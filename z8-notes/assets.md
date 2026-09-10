# Assets — HP Z8 G4 `berlin-3eie`

What this machine actually is, measured rather than assumed, and what each
part means for MoE-Infinity. Figures taken 2026-09-10.

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

## Tenancy

The machine is on the **vast.ai marketplace** (currently unlisted). Its
`kaalia` agent is not passive: it prunes Docker images over a budget of 10% of
the Docker disk, runs CPU-bound GPU probes every few minutes, and is kept
alive by a cron watchdog that restarts it every minute if its log goes stale.
Other stacks live here too (`northstream-*`, `caliper-*`, an `actions-runner`).

See `CLAUDE.md` for how to work around all of that safely.
