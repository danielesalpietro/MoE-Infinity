# Assets — HP Z8 G4 `berlin-3eie`

What this machine actually is, measured rather than assumed, and what each
part means for MoE-Infinity. Figures taken 2026-09-10, except the
moe-store-split and RunPod sections (added 2026-09-13) and the GPT-OSS/5060
Ti MXFP4-serving section (added 2026-09-14), each dated within itself.

---

## Compute

**2 × Intel Xeon Gold 6244** @ 3.60 GHz — 8 cores / 16 threads each, **32
threads total**, 49.5 MiB L3. An unusual high-frequency, low-core-count SKU:
good per-core speed, few cores for parallel builds.

**Two NUMA nodes**, and they are not symmetric:

| | CPUs | RAM |
|---|---|---|
| node 0 | 0-7, 16-23 | 112 GB |
| node 1 | 8-15, 24-31 | 129 GB |

This matters because each GPU hangs off a different node (below). A process
feeding the wrong GPU from the wrong node pays a cross-socket hop on **every**
expert fetch.

Cascade Lake, which is why the whole box is **PCIe 3.0** — see below.

## Memory

**235 GB usable DDR4 ECC RDIMM**, 19 modules populated. Mixed parts:
`HMA82GR7CJR4N-WM` rated 2933 MT/s alongside `HMA82GR7AFR4N-UH` rated
2400 MT/s, so **everything runs at 2400 MT/s** — the whole bank clocks to the
slowest module. Replacing the 2400 parts would lift the rest.

For MoE-Infinity this is the tier that matters most and it is already ample:
`HostMemoryPool` takes `HOST_MEMORY_RATIO` (0.8, a compile-time `#define`) of
total RAM → **~188 GB pool**. Every model we are likely to run fits entirely
in it.

## Persistent memory

**Two regions of 270 GB**, `persistence_domain: memory_controller` (i.e. ADR —
writes that reach the controller survive power loss).

| Namespace | Mode | Device | Used as |
|---|---|---|---|
| `namespace0.0` | **sector** | `/dev/pmem0s` | `/boot/efi`, `/` (100 GB ext4), a 151 GB xfs datastore |
| `namespace1.0` | **fsdax** | `/dev/pmem1` | `/mnt/pmem_emh2`, 248 GB xfs, mounted `dax=always` |

Only namespace1 is fsdax; namespace0 is block emulation. **The machine boots
from persistent memory.**

**DAX is currently inert for our workload.** `core/aio/archer_aio_utils.cpp`
opens offload files `O_RDWR|O_CREAT|O_DIRECT` and never mmaps them, and
`O_DIRECT` already bypasses the page cache. PMEM is still the right home for
the offload directory — it is the fastest block device here — but the `dax`
mount option buys nothing until the IO layer learns to mmap.

**What is actually on them (2026-09-11).**

| Path | Size | Used | Free | Holds |
|---|---|---|---|---|
| `/` (`pmem0s2`, ext4) | 100 GB | 20 GB | 74 GB | the operating system |
| `/grastorp/volumes/33579b38…` (`pmem0s3`, xfs) | 152 GB | **3 GB** | **149 GB** | one abandoned, empty `docker` directory dated 23-24 August. Not ours. Mounted from `fstab` by UUID |
| `/mnt/pmem_emh2` (`pmem1`, xfs, `dax=always`) | 248 GB | 230 GB | **19 GB** | 200 GB of models, plus `emh2_pool.bin` — 26 GB, root-owned, 24 August, no process holding it open, **not ours** |

So the squeeze is on one namespace while **149 GB of identical persistent
memory sits idle on the other**. Moving the offload directory or the HF cache
to `/grastorp` is the clean answer to a full `pmem_emh2` — same hardware, same
speed — but that volume belongs to a setup that is not ours, so it is a
decision rather than a free lunch.

## GPUs

| | GPU 0 | GPU 1 |
|---|---|---|
| Model | **RTX 3090** (Ampere) | **RTX 5060 Ti** (Blackwell) |
| Compute capability | 8.6 | 12.0 |
| VRAM | 24576 MiB | 16311 MiB |
| Power limit | 350 W | 180 W |
| PCIe width | **x16** | **x8** |
| PCIe link cap | 8 GT/s (gen 3) | 32 GT/s (gen 5) — **capped to gen 3 by the platform** |
| NUMA node | 0 (CPUs 0-7, 16-23) | 1 (CPUs 8-15, 24-31) |
| UUID | `GPU-43a04d95-…` | `GPU-f8774b4a-…` |

Driver 595.84, exposing CUDA 13.2. They talk to each other over `SYS` — PCIe
plus the inter-socket link. No NVLink, no direct peer-to-peer.

Both idle at 2.5 GT/s (gen 1); that is ASPM power management, not a fault.
They negotiate up under load.

### Numeric formats

| | RTX 3090 | RTX 5060 Ti |
|---|---|---|
| BF16 | native, tensor core | native |
| FP8 (e4m3/e5m2) | **emulated** — dequant to bf16/fp16; storage saving only, no speedup | **native**, dedicated tensor cores |
| FP4 / MXFP4 | no | **native** |
| AWQ / GPTQ int4 | yes | yes |
| INT8 | yes | yes |

The repo already ships the MXFP4 path — `moe_infinity/kernel/mxfp4_gemm.py`,
`moe_infinity/utils/mxfp4.py`, used for GPT-OSS — and
`moe_infinity/utils/quantization.py` handles GPTQ and AWQ
(`SUPPORTED_QUANT_METHODS = {"gptq", "awq"}`; MXFP4 and FP8 have separate
runtime paths).

So the 5060 Ti is the only card here that can run those kernels natively:
**they are dead code on this host today**. And FP8 timings on the 3090 measure
storage, not throughput — there are no FP8 tensor cores on Ampere.

The fused MoE kernels are compiled for **one** CUTLASS arch, so an `sm_86`
image on the 5060 Ti is a runtime kernel failure, not a slowdown. Running both
means two images.

### Selecting a card

Address GPUs as **CDI devices**, not through the legacy runtime:

```yaml
devices:
  - "nvidia.com/gpu=GPU-43a04d95-0acb-c344-96a9-0efd013e32a4"
```

`runtime: nvidia` plus `NVIDIA_VISIBLE_DEVICES` resolves the id against
whichever CDI spec the runtime reaches first, and here that is not dependable:
a vast.ai service writes its own short-lived specs under `/etc/cdi` with hashed
vendor names — a new pair every few minutes, never cleaned up, twelve
accumulated in a day — and each declares **only the 3090**. Selecting the
5060 Ti by UUID that way fails with `unresolvable CDI devices
D.<hash>/gpu=GPU-f8774b4a-…`, naming a vendor that may already have been
deleted. The UUID is not missing: `/var/run/cdi/nvidia.yaml` (`nvidia.com/gpu`)
lists both cards correctly. Addressing the device explicitly pins the lookup to
that spec, isolation holds — the container sees only the card it asked for —
and it runs under plain `runc`, since CDI performs the injection.

### The 5060 Ti is usable today, as a reference

`torch.cuda.get_arch_list()` on the shipped build includes `sm_120`, so **plain
`transformers` runs on the 5060 Ti with the existing `sm_86` image**, no
rebuild. That is enough to use it as a second, independent reference
implementation — which is how the tie-break behind the 44/48 figure on #207 was
confirmed from a different architecture. What needs `CUTLASS_NVCC_ARCHS=120` is
MoE-Infinity's own fused kernels, nothing else. Its 16 GB also bounds what
fits: OLMoE (13 GB) yes; DeepSeek-V2-Lite (30 GB) and Qwen3-30B (57 GB) no.

### MoE-Infinity's own MXFP4 serving on the 5060 Ti (2026-09-14)

The paragraph above is about plain `transformers`; MoE-Infinity's own served
inference is a separate question, and the two kernels involved are not one
thing. Confirmed by running `openai/gpt-oss-20b` (native MXFP4, ~12.9 GB —
not ~9 GB as an earlier estimate from an incomplete shard list said; verify
against the actual file tree, not a partial sum) end to end:

- **The MXFP4 GEMM kernel** (`moe_infinity.kernel.mxfp4_gemm.fused_mxfp4_gemm`)
  already runs on sm_120 in the everyday `sm86-main7007` image
  (`CUTLASS_NVCC_ARCHS=86`) — confirmed with a standalone kernel-launch
  smoke test before trusting it further.
- **The expert-dispatch kernel** (`moe_infinity/distributed/expert_executor.py`,
  `wait_expert()`) does not: `torch.AcceleratorError: no kernel image is
  available for execution on the device`, reached only once real generation
  starts, not at import or model-load time. Different compiled extension,
  different arch coverage, same "no kernel image" symptom as the
  RunPod/Ada finding above — but this one is a real gap, not a binary-
  compatibility question, because `setup.py`'s default flags never include
  `sm_120` unless `MOE_ENABLE_SM120=1` is set at build time, and this image
  never had it set.
- **The project already ships a dedicated fix for this**:
  `docker/Dockerfile.blackwell` — a separate profile, not a flag to bolt
  onto the everyday one. Different base image
  (`nvcr.io/nvidia/pytorch:25.11-py3`, not `pytorch/pytorch:...`), different
  CUTLASS version (v4.4.0, not v3.9.2 — Blackwell-specific templates the
  older CUTLASS lacks), `MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0`,
  `NVTX_DISABLE=1` (sidesteps the missing-NVTX-header gap noted in the
  RunPod section entirely, rather than installing the headers). `.env` had
  already flagged it — "`CUTLASS_VERSION=v4.4.0, CUTLASS_NVCC_ARCHS=120`...
  Change both together or not at all" — read closely only once this gap was
  hit, not before. `compute_80` stays in `setup.py`'s flags regardless of
  `MOE_ENABLE_SM90`/`MOE_ENABLE_SM120`, so an image built this way is not
  Blackwell-only: it should still serve the 3090 too, via the same binary
  compatibility this document already relies on elsewhere. Built once here,
  8 minutes (`docker build -f docker/Dockerfile.blackwell .`), tagged
  `moe-infinity-blackwell:sm120-main7007`.
- **`device_memory_ratio` needs its own number per card, not a value
  carried over from the 3090.** `0.5` (this project's usual OLMoE-era
  default) OOMs on the 5060 Ti's 16 GB; `0.3` loads (62 s, reusing an
  already-materialised `/offload`) and generates (4.5 s) cleanly:
  `"The capital of France is Paris."` Confirmed not just by hand but by the
  repository's own unmodified smoke harness
  (`tests/python/integration/test_model_smoke.py::test_generate_smoke[gpt_oss]`,
  `MOE_GPT_OSS_SMOKE=1`), which hardcodes `0.75` for this model and fails
  with the identical real `cudaMalloc` OOM on this card, run to completion,
  not assumed (1:15:28 wall time — almost entirely the snapshot download at
  100 Mb/s, the test itself takes minutes). `0.75` was tuned on whatever
  card originally earned GPT-OSS its "20B validated" line in
  `docs/model-compatibility.md`; evidently not a 16 GB one, and the harness
  itself does not parametrise the ratio per card.
- A repeat of a lesson from the RunPod section, on a different axis this
  time: a checkpoint download excluding files you don't need
  (`ignore_patterns=[...]`, done once here to skip a duplicate
  `original/model.safetensors`) makes `huggingface_hub` consider that local
  snapshot **incomplete** for any later call that does not pass the same
  `ignore_patterns` — including `MoE()`'s own internal `snapshot_download`,
  which takes none. `HF_HUB_OFFLINE=1` then fails outright instead of
  fetching the gap; without it, `snapshot_download` quietly re-fetches
  whatever was excluded. Either pass identical `ignore_patterns`
  everywhere a given cache is read, or do not exclude anything from the
  original download at all.

## PCIe — the ceiling

**The platform is PCIe 3.0.** Cascade Lake has no gen 4.

| Device | Link | Theoretical |
|---|---|---|
| RTX 3090 | gen 3 **x16** | ~15.75 GB/s |
| RTX 5060 Ti | gen 3 **x8** | ~7.88 GB/s |
| NVMe `nvme0n1` | gen 3 x4 | ~3.9 GB/s |
| NVMe `nvme1n1` | gen 3 x4 | ~3.9 GB/s |

This is the binding constraint for expert offloading. With the RAM tier ample
and the disk quiet after first load, the remaining cost is RAM → VRAM, and it
is capped at gen 3. The 5060 Ti is doubly handicapped: half the lanes, and its
gen-5 capability is unusable here.

The lever is therefore **VRAM residency**, not bandwidth: keep more experts on
the card so the transfer happens less often. Hence
`MOE_DEVICE_MEMORY_RATIO=0.80` (19.2 GB of 24 engaged), and hence int4
AWQ/GPTQ being interesting — at int4 both our checkpoints would be fully
resident.

## Storage

| Device | Size | Type | Filesystem | Role |
|---|---|---|---|---|
| `pmem1` | 248 G | PMEM fsdax | xfs `dax=always` | **`/mnt/pmem_emh2`** — HF cache, expert offload, chat db, git clone |
| `pmem0s` | 252 G | PMEM sector | vfat + ext4 + xfs | `/boot/efi`, `/`, a datastore volume |
| `sdc` | 931 G | SATA HDD | xfs `wdc-docker2` | **`/mnt/wdc-docker`** — Docker data-root since 2026-09-10 |
| `nvme0n1` | 894 G | NVMe (Samsung PM9A3) | ext4 `p2ptest` | GPUDirect Storage tests; currently `/mnt/backup` |
| `nvme1n1` | 477 G | NVMe (Samsung MZVLB) | ntfs | `actions-runner`, `DockerWSL` — in use, not ours |
| `sdd` | 477 G | SATA SSD (Micron) | vfat + ntfs | **Windows install** (dual boot) — do not touch |
| `sdb` | 931 G | SATA HDD | ntfs, Intel RST RAID member | `SteamLibrary`, `Download` |
| `sda` | 466 G | SATA HDD (WD5000AAKS, ~2008) | xfs | **FAILING** — see below |

Measured sequential read: `sda` 119.7 MB/s, `sdc` 131.8 MB/s.

### `sda` is failing

```
ata1.00: error: { UNC }
sd 0:0:0:0: [sda] Sense Key : Medium Error [current]
Add. Sense: Unrecovered read error - auto reallocate failed
I/O error, dev sda, sector 497427392 op 0x0:(READ)
```

"auto reallocate failed" means the drive could not remap the bad sector.
Unmounted, removed from `fstab`, holding the pre-migration Docker copy as a
cold rollback. **Awaiting physical replacement with a 1–2 TB disk.** Exactly
one file was unreadable out of 1,859,628 — a regenerable `.pyc`.

## Network

**`eno1` negotiated at 100 Mb/s.** Measured ~7 MB/s during a HuggingFace pull,
which matches. `00:68:eb:9b:90:0d`, gateway 192.168.1.1, host 192.168.1.110.

This is the biggest avoidable bottleneck on the machine: 31.4 GB takes ~75
minutes at this rate, and the 10 GB PyTorch base image took 27.

**Confirmed 2026-09-14: this 100 Mb/s cap is the NIC's own negotiation, not
the uplink.** A speed test on the same network, wired directly into the
router and bypassing the WiFi access point (a different device from the
Z8, which is itself already wired): 76.13 Mbps down / 8.20 Mbps up, ping
10-77 ms. That is most of a full 100 Mbit link's real-world throughput —
the ISP connection has headroom the Z8 cannot currently reach because its
own port negotiates at 100 Mb/s, full stop, independent of anything
upstream. It settles which half of the network is actually worth fixing:
not the ISP link, the Z8's own NIC/switch port.

**Planned:** 4 x 1 Gbit, replacing the single 100 Mbit link. Worth being
precise about what that buys, because it is not a straight 40x:

- **A single download does not get faster.** LACP and every common bonding
  mode hash per *flow*, so one TCP connection is still capped at 1 Gbit. The
  gain on a HuggingFace pull comes only from `snapshot_download` fetching
  several files concurrently.
- **The internet uplink stays the ceiling** regardless. Bonding helps LAN
  transfers — between hosts here, or to a local registry or cache — not the
  path to the outside.

So the durable lever against the uplink is **not re-downloading**: keeping the
HF cache on PMEM, and keeping it backed up, is worth more than bandwidth. A
31.4 GB checkpoint fetched once and preserved beats any link upgrade.

The public IP is dynamic and moves several times a day, including the second
octet. Reached over SSH on port 2222.

## MoE-Infinity no longer owns model code (the moe-store split)

**As of 2026-09-13 (`main` @ `e51dd67`, upstream PR #216), everything below
this line about "the stack as built" describes a pre-refactor image and is
historical, not current.** `moe_infinity/models/` — every model wrapper and
every `*PagedAttention` shim, 30 modules — is deleted from this repository.
It moved to a separate package, [`moe-store`](https://github.com/EfficientMoE/moe-store),
pulled in via a git pin in `requirements.txt` (currently `@v0.1.0`, a tag —
though a pin can target any commit SHA, a real release is not required to
pick up a merged change). `utils/hf_config.py`, `utils/checkpoints.py`,
`utils/fp8.py` and `common/constants.py` moved the same way, repointed to
`moe_store.parsing.*` / `moe_store.checkpoints` / `moe_store.fp8` /
`moe_store.registry.constants`.

Two things stay on the MoE-Infinity side and do **not** move: the serving
engine (`serving/`, `runtime/`) and `moe_store/wrappers/paged_attention_registry.py`'s
consumer. That registry class itself — layer-bound proxies, per-layer
metadata install/clear — now lives in `moe-store` too, but it is **Qwen3-only
by design** (hardcoded `_QWEN3_MODULE`/`_QWEN3_CLASS`, an alias
`register_qwen3 = register`) and does not need extending for other shims.
Any other shim's metadata installation goes through a separate,
already-generalized, name-based path entirely on the MoE-Infinity side:
`ModelRunner._get_paged_attention_classes()` walks `model.modules()`,
matches `cls.__name__` against a literal set (`_LAYER_REGISTERED_SHIMS` in
`model_runner.py`), and calls `set_paged_context`/`clear_paged_context`
directly — confirmed by tracing every call site of both methods before
relying on it, not assumed from the class's name. Contributing a new shim
therefore means: the shim class itself goes in `moe_store/wrappers/`, wired
into that package's lazy `__init__.py` `__getattr__`; the name goes into
`_LAYER_REGISTERED_SHIMS` on the MoE-Infinity side; nothing touches
`paged_attention_registry.py`.

Practical consequence for any PR open against `main` before this refactor
landed: it will show `CONFLICT (modify/delete)` on
`moe_infinity/models/__init__.py` and, if it also edits
`runtime/model_offload.py`, an auto-merge that is textually clean but
imports a symbol that does not exist yet in `moe-store` — worth a manual
read, not a blind accept of git's merge result.

## The stack as built

| | |
|---|---|
| Serving image | `moe-infinity-serve:sm86-pr195` (`7f4c7650f4a9`, 2026-09-10) |
| Built from | `docker/Dockerfile.serve`, CUTLASS v3.9.2, `CUTLASS_NVCC_ARCHS=86` |
| Carries | our merge of PR #195 onto `main`, plus `num_layers` at `big_modeling.py:483`, plus both halves of #207 (`olmoe: 5` and the bare-tensor return) |
| torch / CUDA | 2.9.1+cu128 / 12.8 |
| transformers | 5.17.0 |
| FlashAttention | not installed (`INSTALL_FLASH_ATTN=false`) |
| FlashInfer | **not installed** — a pip extra in `setup.py`, referenced nowhere in the Dockerfile |

`moe-infinity-serve:sm86` is the unpatched image, kept deliberately as the
`main`-equivalent control. Several findings were established by running one
script against both.

**FlashInfer's absence is not a gap for this workload.** It gates exactly two
things: chunked prefill (`supports_chunked_prefill()` returns
`_flashinfer_enabled()` outright) and prefix KV reuse (`create_layered_store()`
is reached only through that path). Both are multi-request features and do
nothing for one request with a unique prompt. Without it, prefill runs on SDPA
and decode on the compiled `paged_attention_v1` kernel — the arithmetic is the
same. Adding it would cost a second KV store written alongside the first and a
JIT compile per `head_dim` on first use, and MLA is excluded from it by a
`{64, 128}` allowlist regardless.

### Models resident

Each costs its size **twice**: once as the HuggingFace download, once
re-materialised into MoE-Infinity's flat `archer_param_N` partitions — 10 GiB
each, read with `O_DIRECT` — plus a small `archer_index` mapping tensor id to
file and offset.

| Model | Each copy | On disk |
|---|---|---|
| Qwen3-30B-A3B | 57 GB | 114 GB |
| DeepSeek-V2-Lite-Chat | 30 GB | 60 GB |
| OLMoE-1B-7B-0924-Instruct | 13 GB | 26 GB |

The offload half regenerates from the HF cache in minutes; the HF half is a
download, and on a 100 Mbit line Qwen3 took 2h26m. **When space is needed,
delete offload directories, never the cache.**

## What this machine actually delivers

Measured, not inferred.

| | |
|---|---|
| Numerical noise floor | **0.148** logprob spread, run to run, identical request, `temperature=0` |
| Model's own margin at the same position | **0.125** (plain `transformers` on CPU, deterministic across runs) |

The consequence is the part that matters: **the engine's run-to-run noise
exceeds the margin the model is trying to express.** A greedy choice between
two close candidates becomes a coin flip — 17/13 over 30 runs, with 11 exact
ties.

**Withdrawn 2026-09-11.** An earlier version of this section generalised from
that one position to "where the model's margin is wider than the noise, the
engine agrees every time". A second measurement — classic path
(`MoE.generate()`), OLMoE, 432 generations, pre-merge image `sm86` + #207,
`logbook_s002.md` — contradicts it: 6 outliers, of which 4 sit on an exact bf16
tie but **2 crossed real reference margins of 0.50 and 0.25**. The rank
inversion is measured; the amplitude of the noise in those runs is not, because
the native engine returns tokens and no logits. So the generalisation is
withdrawn and no replacement figure is stated. What remains measured: at one
position on DeepSeek serving, a 0.148 logprob spread against a 0.125 reference
margin; on the classic path, the prefill-form forward is bit-exact over 120
repetitions, `torch.use_deterministic_algorithms(True)` does not remove the
jitter (3/144), and plain `transformers` on the same GPU is a point mass
(72/72). All of it predates the 2026-09-11 upstream merges and is to be
re-measured on an image built from `main`.

So on this host **no correctness check may rest on a single run, and none may
assert exact tokens from a GPU run.** Use a fixed N with a stated distribution,
a tolerance on logits, or a CPU reference — the same comparison on CPU is
deterministic.

Throughput, for sizing work rather than for benchmarking:

| Model | s/token | Note |
|---|---|---|
| DeepSeek-V2-Lite | 0.86 | GPU busy at 74%, so not stalling on offload |
| Qwen3-30B-A3B | 12.6 | 57 GB moving through the offload path |

## RunPod — ephemeral GPU verification off this machine

Used 2026-09-13 for two things the Z8's two cards cannot give: a third,
independent GPU architecture, and enough VRAM headroom to push OLMoE
concurrency past this project's previous ceiling of 4. Pod destroyed the
same day (both compute and its network volume) — nothing here describes a
standing resource, only what to expect if one is provisioned again.

**Advertised specs are not the real ones — check twice, from three angles.**
A "16 vCPU / 71 GB RAM" listing, on the L40S pod actually used, resolved to
three different numbers depending on where you look:

| | Advertised (listing) | Naive view inside the container | Real, kernel-enforced |
|---|---|---|---|
| CPU | 16 vCPU | `nproc` / `lscpu`: 256 (the whole physical host — an AMD EPYC 7702, shared, not this pod's allocation) | **27.2 core-equivalents** (`cpu.cfs_quota_us`/`cpu.cfs_period_us`, cgroup v1) |
| RAM | 71 GB | `free -h`: ~1.0 TiB (same host-wide illusion) | **125.0 GB** (`memory.limit_in_bytes`, cgroup v1), confirmed independently by `RUNPOD_MEM_GB=125` in `/proc/1/environ` |

The container's cgroup version was v1, not v2 — the v2 paths
(`/sys/fs/cgroup/cpu.max`) exist as an empty tmpfs and silently return
nothing; the real limits are under `/sys/fs/cgroup/cpu/` and
`/sys/fs/cgroup/memory/`. Use the cgroup-derived number for anything like
`BUILD_JOBS`, never `nproc` — the Z8's own "152 compilers on 32 cores"
incident is exactly what trusting the naive view here would reproduce, and
this host had even more headroom for that mistake (256 apparent cores).
`RUNPOD_API_KEY`/`RUNPOD_POD_ID`/etc. are injected into PID 1's environment
for pod self-management via `runpodctl`, but not inherited by SSH login
shells — read them from `/proc/1/environ` if needed. On this pod the key
itself was rejected by RunPod's API with `Unauthorized` on every call
(`get pod` included) — a dead end unrelated to any local permission, not
retried differently.

**Ada Lovelace (L40S, sm_89, compute capability 8.9) was never explicitly
targeted by this project's build.** `setup.py`'s CUDA arch flags are
`compute_80` (always), `compute_90` (`MOE_ENABLE_SM90`, default on),
`compute_120` (`MOE_ENABLE_SM120`, opt-in) — no `sm_89` anywhere, confirmed
by grep before relying on it. It ran anyway: CUDA's binary-compatibility
guarantee (a cubin built for compute capability X.y runs on any device X.z
with z≥y, same major version) covers `sm_80` running on `sm_89`, and the
Z8's own RTX 3090 (sm_86) is a live instance of the same fact — `setup.py`
never targets `sm_86` explicitly either, and every measurement on this
project's Z8 GPU has depended on that same compatibility path without
incident. Confirmed here too with a dedicated kernel-launch smoke test
(one real paged-attention forward, GPU-only) before spending time on full
suites — worth repeating on any future Ada/Hopper-class pod before trusting
the build. Set `MOE_ENABLE_SM90=0` on an Ada-only pod: it halves the CUDA
compile matrix for no benefit (nothing here runs the sm_90 cubin), at the
cost of `moe_store`'s `_v4fp4_arch_flags` falling back to `sm_120a`-only
(Blackwell) for the FP8/MXFP4 extension specifically — harmless unless that
extension is what's under test (GLM-5.x, GPT-OSS), which it was not here.

**Build recipe: CUDA 12.8, not whatever the pod defaults to.** Pick 12.8 at
pod creation when offered a choice — it is the version `docker/Dockerfile`
and the README's "Install from Source" section actually use
(`pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel`, `pip install torch
--index-url .../whl/cu128`); anything else is unverified territory this
project has never built against. The README is missing a real step the
Dockerfile has: NVTX C++ headers
(`git clone --depth 1 --branch v3.2.2 https://github.com/NVIDIA/NVTX.git`,
copy `c/include/nvtx3/` into the CUDA include path) — `nvtx3.hpp` is
included directly by `core/aio/archer_prio_aio_handle.cpp` and three other
`.cpp` files; skip it and the build fails on those. Ubuntu 24.04's system
Python refuses plain `pip install` (PEP 668); `python3 -m venv
--system-site-packages` reuses the image's pre-installed, already-matching
torch instead of redownloading it, while keeping `pip install` usable.

**Ephemeral vs persistent, on a RunPod pod specifically**: only what's under
the mounted network volume (`/workspace` here) survives a stop; everything
else — apt packages, the NVTX headers just installed into
`/usr/local/cuda/include`, any environment variable — is gone and must be
redone each time the pod (re)starts. `CUTLASS_DIR` and any manually-cloned
build dependency belong under the persistent mount too, not under `~`.

**Result, for the record**: OLMoE with this project's three engine fixes
(PR #213/#217), teacher-forced against plain `transformers`, agreement
≥0.99 on this architecture, sequential and concurrent up to **32**
simultaneous requests (`max_batch_size` raised to match — the engine's
default cap of 8 would otherwise chunk a nominally-32-wide batch into
smaller ones without saying so).

## Working on this machine

- **The Z8 has no GitHub credentials, deliberately** — shared host, public
  repository. The flow is: commit on the Z8, `git format-patch -1 --stdout`,
  `git am` on the Windows checkout, push from there, then
  `git reset --hard origin/<branch>` on the Z8. Check that the working file is
  unchanged by the reset before trusting it.
- **Every measurement carries `~/moe-infinity/declare-env.sh`**, run *before*
  the run starts, not during. It records host and driver, branch and commit,
  image id and build args, **which patches are actually present inside the
  container** — checked one by one rather than assumed — the run configuration,
  T0 for GPU/CPU/RAM/disk, and the background noise: other containers, `cron`,
  `vastai`, anything over 10% CPU. Snapshots are archived beside the logbook.
- **The Z8 doubles as a verification host for code destined elsewhere** —
  used 2026-09-13 to check formatting and run the test suite for a change to
  the separate `moe-store` package while its own GitHub Actions sat on
  `action_required` (a first external contribution to that repository gates
  on manual maintainer approval, not something this project or any
  automation can push past). Pattern: a fresh, timestamped directory under
  `/tmp` (never reused — see the worktree rule below), a plain `git clone`
  of whatever branch, `docker run -v <dir>:/repo -w /repo <matching base
  image> bash -c '<the target repo's own CI commands, verbatim>'` — reading
  the target's `.github/workflows/*.yml` first to match what its CI actually
  runs, not guessing. Files written by the container land root-owned; clean
  up through another container (`docker run --rm -v <dir>:/x <image> rm -rf
  ...`), and remember `rm -rf dir/*` silently skips dotfiles — `.ruff_cache`,
  `.pytest_cache` need an explicit pattern or they survive as orphaned
  root-owned litter on a shared host.

## Tenancy

The machine is on the **vast.ai marketplace** (currently unlisted). Its
`kaalia` agent is not passive: it prunes Docker images over a budget of 10% of
the Docker disk, runs CPU-bound GPU probes every few minutes, and is kept
alive by a cron watchdog that restarts it every minute if its log goes stale.
Other stacks live here too (`northstream-*`, `caliper-*`, an `actions-runner`).

See `CLAUDE.md` for how to work around all of that safely.
