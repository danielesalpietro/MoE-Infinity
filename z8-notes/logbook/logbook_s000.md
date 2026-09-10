# Logbook s000 — 2026-09-09 (UTC)

Host: `berlin-3eie` (HP Z8 G4) · repo `danielesalpietro/MoE-Infinity`, branch `feat/z8-serve-webui`

Days are counted in **UTC**, the Z8's own clock, so entries line up with build
logs, journals and the kaalia log they cite.

---

## Objective

Stand up MoE-Infinity on the Z8 so the OLMoE inference crash
(EfficientMoE/MoE-Infinity#123) can be reproduced on real hardware. Nobody on
the upstream thread has exercised the fused CUDA path — both positions rest on
source reading and CPU-side shape checks — so a live run is the missing
evidence.

## Done

- **SSH restored.** Timeout was a dynamic public IP, not a dead machine.
- **Branch `feat/z8-serve-webui`** off current `upstream/main` (`02a4afe`).
  Serving infra ported from the fork's 2-month-old `feat/openai-webui-serve`;
  its library changes deliberately left behind, since diffing them against
  current main showed they would revert upstream work (GLM-MoE-DSA, Qwen3.5,
  CUDA graphs, chunked prefill).
- **Two upstream issues opened** on the fork: #2 (RunPod deployment package),
  #3 (Blackwell sm_120 profile for the RTX 5060 Ti).
- **OLMoE checkpoint cached**: `allenai/OLMoE-1B-7B-0924-Instruct`, 13 GB, on
  PMEM. Config verified against the claim in #123 — `hidden_size` 2048,
  `intermediate_size` 1024, bfloat16. The inequality is what makes the
  `H == H_out` check fail rather than silently computing wrong outputs.
- **Isolated Docker daemon** (`docker-moe` + `containerd-moe`), with
  install/check/uninstall scripts in `~/moe-infinity/`.

## Incidents

### vast.ai deleted the first build

The first image was built on the system daemon and removed 39 seconds after it
was tagged:

    update_image_cache diskspace:465.5GB maximg_space:46.6GB totimg_space:53.1GB
      removing moe-infinity-serve:sm86: af6917b1...
    result: Untagged: moe-infinity-serve:sm86

kaalia enforces a fixed image budget of 10% of the disk and enumerates via
`docker images -q` on the system socket. **The isolation that matters is the
socket, not the filesystem** — a symlink or a different mount would not have
helped, because deletion happens through the API.

An hour of build lost. Cause understood, not repeated.

### Our containerd hijacked the host socket

`containerd` 2.3 moved the gRPC server into a plugin and **ignores a top-level
`[grpc]` table in a version-4 config**. Our instance did not fail — it came up
on the containerd default, which is `/run/containerd/containerd.sock`, the
*system* socket. It unlinked and rebound that path, and on shutdown deleted it.

The system dockerd kept answering on an already-established connection, so an
API-level health check reported everything fine while every new client was
broken. Recovered with `systemctl restart containerd.service`.

Fixed by passing `--address`, `--root`, `--state` as **command-line flags**,
which take precedence and do not depend on config schema. `01-install` now
fingerprints the system socket's inode before and after starting ours and
aborts and restores if it moved.

### Port collisions with the other stacks

3000, 8000 and 8600 were all already claimed — by `northstream-open-webui`,
`northstream-landing` and `caliper-verifier`, all stopped but all carrying
`restart: unless-stopped`. Nothing would have failed today; those stacks would
have failed to start later, with no obvious connection to this work. Moved to
8700-8702.

### Leaked image found by the check

`moe-infinity-model-status:latest` was sitting in the system daemon, left from
the pre-isolation build. Removed after confirming no container referenced it.

### The build starved vast.ai's health check

From the vast.ai console:

    bad bandwidthtest2: {'ERROR_CONDITION': 'not r_H2D or not r_D2H or not r_D2D', 'gpu_idx': 0}
    Your machine is no longer visible in search due to an error.

Not hardware, and not the containerd restart. `docker/Dockerfile.serve` builds
CUTLASS with `cmake --build ... -j$(nproc)` and no ceiling: on 32 cores that
produced 152 concurrent compiler processes and a load average of 32.75.

kaalia's probe is `docker run --rm --runtime=nvidia ... bandwidth-test-nvidia
--csv --device=0`. H2D and D2H are pinned-memory copies, so they are
CPU-bound; under full saturation the run stretched from seconds to ~90-100s
and came back missing rows for one of the three directions, which is exactly
what `not r_H2D or not r_D2H or not r_D2D` reports.

The correlation is clean in the kaalia log — results alternate pass/fail only
from the moment compilation started:

    21:47:20  bandwidthTest2 0
    21:55:43  bandwidthTest2 1
    21:56:57  bandwidthTest2 0
    22:04:39  bandwidthTest2 1
    22:06:34  bandwidthTest2 0
    22:13:35  bandwidthTest2 1

GPUs were idle and healthy throughout (P8, 32-36 C, 0% util, no ECC or Xid
errors), and the result payload was still being sent — vast.ai's backend was
rejecting the contents, not failing to receive them.

**Decision: let it run.** Self-limiting; the listing recovers when the build
ends. Mitigations considered and not taken: `CPUQuota=2400%` on
`docker-moe.service` (applies live via cgroup v2, leaves 8 cores free, costs
~25% build time), or stopping `vastai.service` for the duration.

**Worth fixing properly:** `-j$(nproc)` is the wrong default on a machine
rented out to other people. A `BUILD_JOBS` build arg on `Dockerfile.serve`,
defaulting to something below `nproc`, would make this stack a good tenant
instead of one that knocks the host out of the marketplace on every rebuild.

## Corrections to earlier decisions

- **Image store moved off PMEM** to `/mnt/wdc-docker/docker-moe`, on the user's
  challenge. Isolation comes from the socket, so PMEM bought nothing, and image
  layers are read once at container start. PMEM is reserved for the workload
  (HF cache, expert offload, chat database).
- **Check script rewritten twice.** It compared image IDs and flagged
  `ubuntu:22.04` appearing in both daemons as a leak — content-addressed IDs are
  identical wherever you pull from, so that was a false positive. The real
  assertion is that no `moe-infinity-*` image is visible to the system daemon.
- **GPU check tested the first card, not the configured one.** A stack pinned
  to GPU 1 would have got a green light from a test that exercised GPU 0. Now
  reads `MOE_GPU_DEVICE_IDS` from `.env` and also asserts the compute
  capability matches `CUTLASS_NVCC_ARCHS`.

## Open at end of day

- **Build running** in the isolated daemon since 21:40, step #11 (base image
  pull). Survived a TCP reset at 22:01:46 caused by the same network event that
  changed the public IP; BuildKit retried and resumed.
- **Not yet done**: the three-phase reproduction (`docker/repro_olmoe_123.sh`)
  — smoke test on tiny-random-Mixtral, then OLMoE with `olmoe = 4` expecting
  the crash, then the flip to 5.
- **Promised upstream, not opened**: the OLMoE 4→5 PR with the one-line test
  update. Deliberately held until there is runtime evidence.
- **Undecided**: whether to add vast.ai's registry mirrors to the isolated
  daemon. Omitted for hygiene, but direct pulls from CloudFront are slow and
  flaky on this line. Mirrors are unrelated to image pruning, so it would be
  safe.
- **Worth fixing**: the public IP changed three times in a few hours, across
  different second octets, with no reboot. A DHCP reservation or dynamic DNS
  would remove this friction.
- **Worth fixing**: `BUILD_JOBS` build arg, so a rebuild does not delist the
  host from vast.ai (see the health-check incident above). **No issue for
  this** — see the convention below.

## Per-problem tracking

Issue #123 has its own file that spans days: **`logbook_issue123.md`**. The
daily logbooks say what happened on a date; that one says where the
investigation stands. The reproduction was confirmed on this date — see it for
the evidence and the remaining steps.

## Conventions agreed today

- **GitHub issues are for MoE-Infinity itself, not for Z8 operations.**
  Anything about this host — the isolated daemon and its scripts, build
  ergonomics on this machine, vast.ai friction — belongs in this logbook. Only
  changes that stand on their own as project work get an issue.
- **Logbook days are counted in UTC**, the Z8's own clock, so entries line up
  with the build logs, journals and kaalia log they cite. `new-day.sh` derives
  the counter from the highest existing file, so it cannot drift.
- **`CLAUDE.md` in the repo working copy on the Windows box** carries the
  standing context for a new session: environment, decisions taken, and actions
  decided but not yet executed. It points here for the running record. Kept
  **git-ignored** — it describes this specific host (vast.ai tenancy, storage
  layout, LAN address) and the fork is public.
