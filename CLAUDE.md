# Project context for a new session

Read this first, then the latest logbook entry. Everything below is the state
as of **2026-09-10 (UTC)**. Where this disagrees with a logbook, the logbook
is right and this file is stale.

---

## What we are doing

Running MoE-Infinity on a real GPU to settle the OLMoE inference crash,
[EfficientMoE/MoE-Infinity#123](https://github.com/EfficientMoE/MoE-Infinity/issues/123),
with evidence rather than argument. The upstream discussion on
[#205](https://github.com/EfficientMoE/MoE-Infinity/pull/205) had two people
reasoning from source and CPU-side shape checks — **neither had executed the
fused CUDA path**, and the `TORCH_CHECK` that fires is in device code.

**The claim:** `MODEL_MAPPING_TYPES["olmoe"] = 4` routes OLMoE through the
Mixtral branch of `MoEMLP::ForwardHelper`, which reads the expert weight blob
as `[gate, down, up]`. OLMoE's HF module registers
`[gate_proj, up_proj, down_proj]`, so up and down arrive swapped and
`fused_moe_ffn_into` gets a `down_proj` of the wrong shape.

### Where it stands (2026-09-10)

| | |
|---|---|
| Crash reproduced on hardware | **yes** — engine dies unprompted, no request needed |
| `olmoe` 4 → 5 removes it | **yes** — 0 occurrences after the patched restart, 2 before |
| OLMoE serves end to end | **no** — a second, independent bug is underneath |
| Positive control | **missing**, being fetched |

The flip does not *cause* the second bug, it **uncovers** it: with `olmoe = 4`
the kernel check aborted upstream of the failing line.

```
transformers/models/olmoe/modeling_olmoe.py:402  hidden_states = residual + hidden_states
TypeError: unsupported operand type(s) for +: 'Tensor' and 'tuple'
```

`moe_infinity/models/olmoe.py` returns `(hidden_states, router_logits)` — the
transformers v4 contract — while v5's decoder layer consumes `self.mlp(...)`
as a bare tensor. Static comparison says mixtral, jamba, dbrx, gpt_oss and
nllb_moe return tuples too; **only OLMoE has been run**, so the scope is
unverified.

**The open weakness:** there is still no model that demonstrably serves on this
stack, so no failure can be attributed with full confidence. The intended
smoke test, `vprovorg/tiny-random-Mixtral-8x7B-v0.1`, is useless for it —
random weights with degenerate dimensions (6×16), generation dies inside
torch's own `F.linear`. `deepseek-ai/DeepSeek-V2-Lite-Chat` is being fetched
instead, chosen on a falsifiable prediction: its wrapper returns a bare
tensor, so it should serve. If it fails the same way, the contract analysis
above is wrong — and that has to be said on #206, where it was promised.

## Where the memory lives, and which file does what

Four kinds of document, each answering a different question. Writing something
in the wrong one is how it gets lost.

| File | Answers | Shape |
|---|---|---|
| **`CLAUDE.md`** (this file) | *What is true now?* | Standing state. Rewritten in place when reality changes — never appended to. If it disagrees with a logbook, the logbook is right and this file is stale. |
| **`todolist.md`** | *What do we owe?* | A queue plus the gate rule. Items move out when done, not down. |
| **`assets.md`** | *What is this machine?* | Measured hardware inventory and what each part implies. Rewritten when the hardware changes. |
| **`logbook_sNNN.md`** | *What happened on that day?* | One file per UTC day, append-only. Never edited afterwards — a corrected logbook is a lost lesson. |
| **`logbook_issue123.md`** | *Where does that investigation stand?* | Per-problem, spans days, updated in place. The daily logs say what happened; this says what it means. |

Locations:

| | |
|---|---|
| All of the above | Z8: `~/moe-infinity/` and `~/moe-infinity/logbook/` |
| Next-day helper (counter derived, never typed) | Z8: `~/moe-infinity/logbook/new-day.sh` |
| Retired scripts, with a note saying why not to run them | Z8: `~/moe-infinity/archive/` |
| Third copy of the logbooks, the todo, `assets.md` and `declare-env.sh` | this repo: `.z8-logbook/`, excluded via `.git/info/exclude` |
| Cross-session facts about the user and the project | `~/.claude/projects/<this-project>/memory/` (see `MEMORY.md`) |

**Reading order for a new session:** this file, then `todolist.md`, then the
highest-numbered logbook, then `logbook_issue123.md` if the OLMoE work is what
you are picking up. `logbook_s000.md` is worth reading even when stale —
most of it is *why* things are the way they are, and that does not expire.

**And one document that is not ours:** `docs/model-compatibility.md`, in the
upstream tree. It is the project's own record of what is validated, what is
merely implemented, and what has no evidence at all — per family and per
capability. It is checked into every branch we work on, it changes (last
touched 2026-09-04 by #175), and it must be **re-read and verified against the
current tree**, not remembered. See the convention below.

**Log everything.** Not a nicety: this project has repeatedly been saved by a
timestamp or a log line written down at the time. Incidents go in the logbook
**including our own mistakes**, with the cause, because the mistakes are the
part that would otherwise be repeated.

## The three machines

**This box (Windows)** — the git working copy, at
`C:\Users\salpietrod\OneDrive - WeAreProject\Documenti\Github\MoE-Infinity`.
Two remotes: `origin` = `danielesalpietro/MoE-Infinity` (the fork, public),
`upstream` = `EfficientMoE/MoE-Infinity`. Local `main` is far behind upstream —
branch off `upstream/main`, never off the fork's `main`.

**Working branch**: `feat/z8-serve-webui`.

**The Z8 G4** (`berlin-3eie`) — Ubuntu 24.04, 32 cores, 235 GB RAM, the lab host.

```
ssh -i "C:\Users\salpietrod\Downloads\.ssh\id_ed25519" -p 2222 admin@<IP>
```

**Its public IP is dynamic and moves several times a day**, including the
second octet. A timeout is almost always that, not a dead machine — ask the
user for the current address. The host key is stable and is how you confirm
it is the same machine after a change:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICEb5j7doMBklgLI/UFXw8h6FFM0Z8yauEk4irMHTN/P
```

If the key differs, **do not accept it** — a recycled dynamic IP can land on
someone else's host.

## The Z8 environment, and why it is awkward

The machine is **rented out on the vast.ai marketplace**. Its `vastai.service`
daemon (`kaalia`) is active and `enabled`, and it is not a passive observer:

- It **deletes images**. It polls `docker images -q` on the system socket and
  enforces a fixed budget of ~46.6 GB (10% of the disk). It removed our
  30+ GB serving image 39 seconds after the build tagged it. Deletion is via
  the **API**, so moving files or symlinking storage does nothing.
- It **runs GPU health probes** every few minutes. They are CPU-bound, and a
  saturated machine makes them fail and delists the host from search.

**There is one Docker daemon — the system one. Use it plainly, no
`DOCKER_HOST`, no context.** Anything you read elsewhere about a second
isolated daemon is history; see below.

The image budget is no longer the problem it was. It is 10% of the Docker
disk, and on 2026-09-10 the data-root moved from a 466 GB disk to a 931 GB
one, so the ceiling went from 46.6 GB to **93.1 GB** against ~43 GB in use.
Our 10.9 GB image fits with ~50 GB to spare.

### What not to repeat

For about a day this project ran a **second, isolated Docker daemon**
(`docker-moe` + `containerd-moe`) to keep images out of kaalia's reach. It
worked, and it was removed. **Do not rebuild it.** Two dockerds on one host
are not safe under Docker 29: its nftables rules live in a host-global
`docker-bridges` table, so the second daemon's startup cleanup destroyed
`docker0` and left the system daemon unable to start *any* container — for
hours, silently, because nothing was running to notice.

The archived scripts and the reasoning are at
`~/moe-infinity/archive/isolated-daemon-2026-09/`.

Four things learned there that are still true:

- **`systemctl stop vastai` holds for about two minutes.**
  `/etc/cron.d/vastai_restart_everything` runs every minute and restarts it
  whenever `kaalia.log` goes stale. To quiesce: **stop `cron` first**, then
  `vastai`. Restore in the reverse order.
- **`systemctl mask` does not work on `vastai.service`** — it is a real file
  in `/etc/systemd/system/` and systemd refuses to mask over it.
- **Quiesce before touching mounts.** During a mount transition kaalia reads
  `diskspace:-0.0GB`, collapses its budget to 16 GB and starts pruning. That
  is how an 18.3 GB image belonging to another workload was destroyed.
- **`containerd` alone does not touch nftables; only `dockerd` does.** That is
  what made it safe to start `containerd-moe` on its own to export the image
  with `ctr` before deleting the isolated store.

### Commands — everything runs on the Z8, over SSH

One daemon, so nothing to select:

```bash
cd /mnt/pmem_emh2/MoE-Infinity        # the git clone lives on PMEM
docker ps
```

**Quiescing the host**, before anything that touches mounts or images — the
order matters, cron is the watchdog:

```bash
sudo systemctl stop cron.service     # first: it restarts vastai every minute
sudo systemctl stop vastai.service
pgrep kaalia | wc -l                 # must be 0 before you proceed
# ... work ...
sudo systemctl start vastai.service
sudo systemctl start cron.service
```

**The stack** — `docker-compose.webui.yml`, config from `.env` (copied from
`.env.z8.example`, which documents why each value differs from the defaults):

```bash
docker compose -f docker-compose.webui.yml build
docker compose -f docker-compose.webui.yml up -d
docker compose -f docker-compose.webui.yml ps
docker compose -f docker-compose.webui.yml logs -f moe-infinity
docker compose -f docker-compose.webui.yml down
```

A long build should be detached, or an SSH drop takes it with it — the IP
moves several times a day:

```bash
setsid nohup docker compose -f docker-compose.webui.yml build \
  > /tmp/moe-build.log 2>&1 < /dev/null &
tail -f /tmp/moe-build.log
```

Bring the stack up **immediately after a build**: a container referencing the
image is a second line of defence against any prune.

**Reaching it** from a browser on `192.168.1.0/24`:

| | |
|---|---|
| `http://192.168.1.110:8701` | Open WebUI chat |
| `http://192.168.1.110:8700/v1` | OpenAI-compatible API (`/health`, `/v1/models`) |
| `http://192.168.1.110:8702` | model-status dashboard |

Open WebUI grants admin to whoever registers first — create the account on
first start.

**The experiment** — `docker/repro_olmoe_123.sh`, run from the repo root:

```bash
docker/repro_olmoe_123.sh before    # expect the fused-kernel crash
docker/repro_olmoe_123.sh patch     # flip olmoe 4 -> 5 in the container, restart
docker/repro_olmoe_123.sh after     # expect a completion
docker/repro_olmoe_123.sh report    # host, GPU, image, commit + both logs
```

Evidence lands in `./.repro-123/` (git-ignored). Switch models without
rebuilding by setting `MOE_MODEL` in `.env` and re-running `up -d`.

**Logbook**:

```bash
~/moe-infinity/logbook/new-day.sh --show   # current and next entry
~/moe-infinity/logbook/new-day.sh          # create the next UTC day
```

### Storage

| Path | Holds | Free |
|---|---|---|
| `/mnt/wdc-docker` | the Docker image store, on **sdc1** since 2026-09-10 | ~703 GB |
| `/mnt/pmem_emh2` | **the workload**: HF cache, expert offload, chat db, and the git clone | ~192 GB |
| `/mnt/backup` | `nvme0n1p1`, mounted by hand for this work. **Not in fstab** — a reboot returns it to the GPUDirect Storage tests it belongs to | ~811 GB |
| `sda` | **failing** — unrecoverable read error at sector 497427392, `auto reallocate failed`. Unmounted, out of fstab, awaiting physical replacement. Holds the pre-migration copy as a cold rollback | — |

PMEM is for the offload workload — that is why this machine is interesting for
this project. Image layers do *not* go there; they are read once at container
start.

### Memory tiering — where the levers actually are

Established by reading the code, and it redirects the obvious instincts:

- **The host RAM tier is already large.** `HostMemoryPool` sizes itself at
  `HOST_MEMORY_RATIO` of total RAM — **0.8, a compile-time `#define`** in
  `core/memory/memory_pool.h`, not a runtime flag. On 235 GB that is a ~188 GB
  pool, so every model we are likely to run fits entirely in RAM and the
  offload directory falls quiet after the first load.
- **DAX is inert here.** `/mnt/pmem_emh2` is mounted `dax=always`, but
  `core/aio/archer_aio_utils.cpp` opens offload files
  `O_RDWR|O_CREAT|O_DIRECT` and never mmaps them — and `O_DIRECT` already
  bypasses the page cache. PMEM remains the right disk for the offload
  directory; the DAX flag specifically buys nothing until the IO layer learns
  to mmap.
- **The lever that matters is VRAM.** With the RAM tier ample and the disk
  quiet, the remaining cost is RAM → VRAM over PCIe, and the way to pay it
  less often is to keep more experts resident. `MOE_KV_CACHE_RATIO` is a
  fraction *of* `MOE_DEVICE_MEMORY_RATIO`, not of the remainder, so raising
  the first raises both. Currently **0.80** — 19.2 GB engaged of 24, ~16.3 GB
  experts, ~4.8 GB left for the CUDA context, fused-kernel buffers and
  activations.
- Three undocumented runtime knobs exist for the caching allocator:
  `MOEINF_GPU_SIZE` (defaults to total VRAM), `MOEINF_SHM_SIZE` and
  `MOEINF_PIN_SIZE` — **the last two have no defaults and are `DLOG_FATAL`**
  if their allocator starts without them.

**Never run `docker system prune -a` or `docker volume prune` against the
system daemon.** They are host-wide and would destroy the `northstream-*` and
`caliper-*` stacks that live on this host.

### GPUs — two architectures, not interchangeable

| | GPU 0 | GPU 1 |
|---|---|---|
| | RTX 3090 | RTX 5060 Ti |
| Compute capability | **8.6** | **12.0** |
| VRAM | 24 GB | 16 GB |
| PCIe | x16 | x8 |
| NUMA node | 0 (CPU 0-7,16-23) | 1 (CPU 8-15,24-31) |

### Numeric formats — what each card can actually do

| | RTX 3090 (sm_86) | RTX 5060 Ti (sm_120) |
|---|---|---|
| BF16 | native, tensor core | native |
| FP8 (e4m3/e5m2) | **emulated** — dequant to bf16/fp16, storage saving only, no speedup | **native**, dedicated tensor cores |
| FP4 / MXFP4 | no | **native** |
| AWQ / GPTQ int4 | yes | yes |
| INT8 | yes | yes |

The repo already ships the MXFP4 path — `moe_infinity/kernel/mxfp4_gemm.py`,
`moe_infinity/utils/mxfp4.py`, used for GPT-OSS — and
`moe_infinity/utils/quantization.py` handles GPTQ and AWQ
(`SUPPORTED_QUANT_METHODS = {"gptq", "awq"}`; MXFP4 and FP8 have separate
runtime paths).

Three things follow:

- **The 5060 Ti is not just "the other GPU".** It is the only card here that
  can run those MXFP4 kernels natively, so they are dead code on this host
  today. That is a stronger argument for
  [#3](https://github.com/danielesalpietro/MoE-Infinity/issues/3) than "we own
  a second card".
- **FP8 experiments on the 3090 measure storage, not throughput.** There are
  no FP8 tensor cores on Ampere; it dequantises. Any timing comparison there
  is measuring the wrong thing.
- **AWQ/GPTQ int4 is the portable lever.** It runs on both, and it attacks the
  real constraint. At int4, OLMoE (13.8 GB bf16) is roughly 4 GB and
  DeepSeek-V2-Lite (31.4 GB) roughly 9 GB — both comfortably resident in the
  19.2 GB the 3090 currently engages, which would take the PCIe transfer path
  out of the picture almost entirely.

Worth keeping in proportion: MoE-Infinity's value is the CPU/NVMe offload of
experts, not FP8 acceleration in itself. Whatever the dtype, VRAM is the
binding constraint — the formats matter because they change how many experts
stay resident, not because of arithmetic speed.

### Arch matching is not optional

The fused MoE kernels are compiled for **one** CUTLASS arch. Pointing an
`sm_86` image at the 5060 Ti is a runtime kernel failure, not a slowdown. The
stack is pinned to the 3090 by UUID in `.env`. Nothing enforces the match any
more: the check that asserted the card compute capability against
`CUTLASS_NVCC_ARCHS` lived in the archived `02-check-daemon.sh`. Verify it by
hand when changing GPU or image.

## Decisions already taken

- **One Docker daemon, the system one** (2026-09-10). The isolated daemon was
  built, used, and removed; see "What not to repeat" above.
- **Docker data-root moved from `sda` to `sdc`** (2026-09-10). `sda` is a 2008
  WDC WD5000AAKS that developed an unrecoverable read error. The move also
  doubled kaalia image budget, since it is 10% of the disk.
- **`sda` gets no more data** and will be physically replaced with a 1-2 TB
  disk. Kept unmounted meanwhile as a cold rollback.
- **Image store on `/mnt/wdc-docker`, not PMEM.** Image layers are read once
  at container start; PMEM is reserved for the offload workload.
- **Data as bind mounts, never Docker volumes.** HF cache, offload directory
  and the chat database live under `/mnt/pmem_emh2/moe-infinity`, so they are
  daemon-agnostic — they survived the whole daemon migration untouched. Had
  they been Docker volumes each would have needed exporting.
- **Ports 8700 (API) / 8701 (chat) / 8702 (dashboard)**, bound to the LAN
  address `192.168.1.110`. The obvious 3000/8000/8600 are already claimed by
  the `northstream-*` and `caliper-*` stacks on this host.
- **Only the two-line `olmoe` parsing patch** is carried in `hf_config.py`.
  `MODEL_MAPPING_TYPES["olmoe"]` is deliberately left at the buggy `4`, so the
  crash can be observed before the fix is applied.
- **The 5060 Ti stays out** until the OLMoE reproduction is done. Adding a
  second GPU architecture mid-investigation would make any failure ambiguous.
- **Issues are for MoE-Infinity itself, not for Z8 operations.** Anything that
  is about this host — daemon scripts, build ergonomics on this machine,
  vast.ai friction — goes in the logbook, not in a GitHub issue.

## Decided but not yet done

- **The OLMoE 4→5 PR.** Promised publicly on upstream #205 (flip the constant,
  plus a one-line update to `test_parse_expert_type_new_models`). Held
  deliberately until there is runtime evidence — the reproduction below is what
  makes it worth opening. Check whether #205 has merged first, since it changes
  the base.
- **A positive control.** The investigation has no model that demonstrably
  serves, so no failure can be attributed with confidence. The intended smoke
  test, `vprovorg/tiny-random-Mixtral-8x7B-v0.1`, has degenerate dimensions
  (6x16 matrices) and cannot generate at all. Chosen replacement:
  `deepseek-ai/DeepSeek-V2-Lite-Chat`, 31.4 GB — its wrapper returns a bare
  tensor, matching the transformers v5 contract that the OLMoE, Mixtral and
  Jamba wrappers violate. The download never ran: the script targeted the
  system daemon without `DOCKER_HOST` at a moment when `docker0` was broken.
- **Report the second OLMoE bug.** With `olmoe = 5` the fused-kernel crash is
  gone, but the engine then dies on
  `TypeError: unsupported operand type(s) for +: Tensor and tuple` —
  `moe_infinity/models/olmoe.py` returns `(hidden_states, router_logits)`
  while transformers v5 consumes `self.mlp(...)` as a bare tensor. Independent
  of #123, and it deserves its own issue. Scope across the other nine wrappers
  is unverified.
- **`BUILD_JOBS` build arg** on `docker/Dockerfile.serve`. `-j$(nproc)` spawns
  152 compiler processes on this box, saturates all 32 cores and makes vast.ai's
  GPU health probe fail, delisting the machine. It will do this on every
  rebuild.
- **A DHCP reservation or dynamic DNS** for the Z8, to stop losing the session
  to IP changes.
- **Clear `/mnt/backup`** and unmount `nvme0n1p1`, returning it to the
  GPUDirect Storage tests. It is not in fstab, so a reboot does this on its
  own. Everything in it is redundant *now* — but the 11 GB image tar is the
  only thing standing between a mishap and a one-hour rebuild, so not before
  the investigation is confirmed working.
- **Reboot once** as the real test of the new `fstab`. `mount -a` was verified,
  a boot was not.

## Open GitHub items

**Upstream — `EfficientMoE/MoE-Infinity`**

State as of 2026-09-11.

| | | |
|---|---|---|
| [#207](https://github.com/EfficientMoE/MoE-Infinity/pull/207) | PR, **ours**, open | OLMoE expert type 4→5 plus the v5 bare-tensor return. `REVIEW_REQUIRED`, `MERGEABLE`. `mfethe1` ran an independent RED/GREEN control and reported no concerns. We added the numerical validation (44/48 vs plain transformers, the one divergence on an exact bf16 logit tie) and corrected our own caveat in-thread. **The PR body still carries the wrong attribution** — it blames the missing shim alone — and editing it is still owed |
| [#195](https://github.com/EfficientMoE/MoE-Infinity/pull/195) | PR, `drunkcoding`, open | DeepSeek MLA shim *plus* the KV layer-dimension and decode-write repairs. The thread where the real work is. We reported that `num_layers` is never passed at `big_modeling.py:483` (fatal for every non-MLA model), the three-architecture before/after, the memory regression and the concurrency crash; `mfethe1` reproduced on CPU and found the INT8 allocation clobber, which our conflict resolution already fixes |
| [#191](https://github.com/EfficientMoE/MoE-Infinity/issues/191) | issue, `drunkcoding`, open | DeepSeek decode garbage. #195 + the `num_layers` line fixes it on our hardware. No comments; nothing owed here — it closes when #195 lands |
| [#123](https://github.com/EfficientMoE/MoE-Infinity/issues/123) | issue, ours, open | OLMoE crash. Fixed by #207; closes when that merges |
| [#205](https://github.com/EfficientMoE/MoE-Infinity/pull/205) | PR, `drunkcoding`, open | OLMoE parsing. #207 depends on it — without it OLMoE does not load at all |
| [#206](https://github.com/EfficientMoE/MoE-Infinity/pull/206) | PR, `mfethe1`, open | Jamba 4→5. Same two defects as #207. We published a corrigendum retracting the "Qwen3 correct" row |
| [#119](https://github.com/EfficientMoE/MoE-Infinity/pull/119) | PR, ours | **closed** 2026-09-10, superseded by #205 |

`mfethe1` is a peer contributor, not a maintainer — `author_association: NONE`,
zero merged commits. The "Author" badge on #206 means author *of that PR*. Merge
authority sits with `drunkcoding` (99 of ~130 commits) and `lausannel`.

**Fork — roadmap, not blocking:** [#2](https://github.com/danielesalpietro/MoE-Infinity/issues/2) RunPod package, [#3](https://github.com/danielesalpietro/MoE-Infinity/issues/3) Blackwell profile.

## Working conventions

- **Close one before opening another.** No new GitHub issue or PR while one
  of ours is still open, and a close has to stand on its own merits rather
  than be made to earn the next opening. `todolist.md` holds the queue and the
  current gate state. Genuinely urgent findings — data loss, a security
  problem, a wrong result being published — jump it.
- **Issues are for MoE-Infinity itself, never for Z8 operations.** Anything
  about this host — daemon scripts, build ergonomics here, vast.ai friction —
  goes in the logbook. Only changes that stand on their own as project work
  get filed.
- **Say what was not verified.** Every claim published upstream so far
  distinguishes what was executed from what was inferred by reading source.
  That distinction is the reason the runtime evidence carries weight; spend it
  and it is gone.
- **Consult `docs/model-compatibility.md` before any support claim, and check
  it against the tree.** The project keeps its own matrix of what is
  `validated` (has a repository harness), `implemented/experimental` (code plus
  tiny/unit evidence only) and `not recorded` / `not validated` (no direct
  evidence), split by family *and* by capability — general sync/offload is
  scored separately from continuous serving. Read it before saying a model is
  supported, broken, or regressed, and before framing any finding as a defect:
  a crash on a surface the project already marks experimental is a different
  claim from a crash on a validated one. Re-read it each session rather than
  trusting this summary — it is upstream's file, it moves, and a stale
  recollection of it is worse than none.

  Its closing line is the rule itself, and is worth obeying beyond that file:

  > Use `Not recorded` rather than extrapolating hardware, pairing,
  > route-ahead, sampling, rich batching, or paged-cache support from an
  > adjacent capability.

  **Why this is a standing rule and not advice.** On 2026-09-10 this was
  violated four times in one session, each time by asserting from an adjacent
  fact instead of a measurement, and each time the measurement later disagreed:
  Qwen3 was called "correct" on the strength of a tiny fixture where the only
  observation was that output *varied*, and that wrong row was published to a
  third party working on Jamba; the OLMoE serving collapse was reported without
  noting that the matrix already lists OLMoE continuous serving as **not
  validated**, which made a confirmation read as a regression; a concurrency
  crash was reported as a flat defect without noting that DeepSeek-V2
  continuous serving is `implemented/experimental`; and two successive causes
  were proposed in public for a slowdown that the same document explains as
  intentional — "DeepSeek MLA uses the correct PyTorch fallback, not
  FlashInfer acceleration". The file was in the checkout since 2026-09-09 and
  was not opened until the user pointed at it.

  A corollary from the same day: **an unverified mechanism that matters is
  still worth publishing, labelled unverified.** The dead `_write_decode_kv`
  was found on 2026-09-10 and cut from the #195 comment for being unmeasured;
  another contributor found and reported it independently hours later. Silence
  is not the conservative option — mislabelling is.
- Logbook days are counted in **UTC**, the Z8's clock, so entries line up with
  the build logs, journals and kaalia log they cite.
- Commits end with the `Co-Authored-By` trailer; PR bodies end with the Claude
  Code line.
- **Check before assuming.** This host has surprised us repeatedly, and every
  incident in the logbooks was found by a check written to distrust the
  configuration rather than read it back. The corollary earned the hard way: a
  check that cannot tell "the thing failed" from "the check failed" is worse
  than none, because it triggers recovery on its own malfunction. Two rollbacks
  of a working migration were caused by `timeout` being handed a shell
  function.
