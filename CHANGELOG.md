# Changelog

## `feat/openai-webui-serve` (unreleased, branched from `main`)

Adds a Docker-based chat WebUI for interacting with MoE-Infinity models,
separate from the existing build/test Docker images.

### Added

- [`docker/Dockerfile.serve`](docker/Dockerfile.serve) — serving image that
  builds MoE-Infinity's CUDA extensions and starts the OpenAI-compatible
  continuous-batching server (`api_server_v2`). Runtime behavior (model,
  offload dir, memory/KV-cache ratios, batch size, API keys, prefix caching,
  ContextPilot) is configurable via `MOE_*` environment variables without
  rebuilding the image.
- [`docker/serve_entrypoint.sh`](docker/serve_entrypoint.sh) — entrypoint
  script translating `MOE_*` env vars into `api_server_v2` CLI flags.
- [`docker-compose.webui.yml`](docker-compose.webui.yml) — two-service stack:
  `moe-infinity` (built from `docker/Dockerfile.serve`, GPU passthrough,
  port 8000) and `open-webui` (the `ghcr.io/open-webui/open-webui` chat UI —
  the same one used by the NORTHSTREAM lab, here pointed at MoE-Infinity's
  own OpenAI-compatible endpoint instead of Ollama), wired together via
  `OPENAI_API_BASE_URL`.
- `.gitattributes` — forces LF line endings on `*.sh` files so
  `serve_entrypoint.sh` doesn't get checked out as CRLF on Windows and break
  inside the Linux container.
- README: new "Chat WebUI (Docker)" section under "OpenAI-Compatible Server"
  documenting how to start the stack and override the served model.
- [`start-webui.ps1`](start-webui.ps1) / [`start-webui.sh`](start-webui.sh) —
  launch scripts wrapping `docker compose -f docker-compose.webui.yml up -d
  --build`, with an optional model override argument, mirroring
  NORTHSTREAM's `start-addon.ps1`/`.sh` pattern.
- [`check-webui-status.ps1`](check-webui-status.ps1) / [`check-webui-status.sh`](check-webui-status.sh) —
  monitoring script for the stack once it's running: container status,
  HuggingFace cache size (proxy for model download progress), `/offload`
  dir size, `/health` endpoint, and the last 15 log lines from
  `moe-infinity-server`. Useful during first-time model loading, which can
  take a long time and doesn't otherwise expose progress.
- [`check-system-requirements.ps1`](check-system-requirements.ps1) / [`check-system-requirements.sh`](check-system-requirements.sh) —
  pre-flight check to run before `start-webui`: Docker/Compose, CPU cores,
  RAM, GPU/VRAM, WSL2 memory limit (`.wslconfig`, Windows only), and free
  disk space, checked against min/recommended thresholds (4/8 cores,
  16/32GB RAM, 8/16GB VRAM, 40/80GB free disk) derived from this session's
  build of `deepseek-ai/DeepSeek-V2-Lite-Chat` (~30GB download). Exits
  0/1/2 for ready/warnings/failures.
- [`model-status/`](model-status) — new read-only FastAPI service
  (`docker-compose.webui.yml`, port 8600) showing which models are cached
  locally (size on disk, download-complete status), whether HuggingFace Hub
  has a newer commit than what's cached, and, for any HF repo id looked up,
  its download size plus a pass/warn/fail check against this host's RAM,
  free disk, and VRAM. The compatibility thresholds are heuristics
  calibrated from the RAM-exhaustion crash-loop hit in this session
  building `deepseek-ai/DeepSeek-V2-Lite-Chat` (~30GB) vs.
  `allenai/OLMoE-1B-7B-0924-Instruct` (~13GB, ran cleanly). Mounts the
  shared `moe_hf_cache` volume read-only and never touches the
  `moe-infinity` container -- it only prints the `MOE_MODEL=... 
  ./start-webui.sh` command to run.
- [`.env.example`](.env.example) — documents all `MOE_*`/`HF_TOKEN`/
  `INSTALL_FLASH_ATTN` variables used across `docker-compose.webui.yml`;
  copy to `.env` (already gitignored) and `docker compose` picks it up
  automatically instead of needing them exported by hand each time.
- `--log-level` flag on `api_server_v2.py` (default `info`, matching the
  previous hardcoded value), wired to both `logging.basicConfig` and
  `uvicorn.run`. Set via the new `MOE_LOG_LEVEL` env var (empty/unset by
  default -- deliberately opt-in per run rather than always-on, since
  `debug` is noisy) when diagnosing a slow/stuck model load.
- `cap_add: [SYS_PTRACE]` on the `moe-infinity` service so `py-spy dump
  --pid 1` can be run inside the container for a live Python stack trace --
  used during this session to tell a genuinely deadlocked load apart from a
  slow-but-working one (`/proc/1/task/*/wchan` was the fallback when
  `py-spy` wasn't yet installed/permitted).
- **Dashboard** in `model-status` (same service, same URL): RAM (host
  total, RAM assigned to Docker's WSL2 VM, RAM in use) and disk (host
  free/used, total size of this stack's Docker volumes) as SVG donut
  gauges; a live status/CPU/memory table for `moe-infinity-server`,
  `open-webui`, `model-status`, and `docker-proxy`; and an auto-refreshing
  log viewer. Backed by a new `docker-proxy` service
  ([`tecnativa/docker-socket-proxy`](https://github.com/Tecnativa/docker-socket-proxy))
  that exposes only read-only `GET` Docker API calls (`CONTAINERS=1`,
  `INFO=1`, `POST=0` -- no exec/start/stop/create), which `model-status`
  additionally restricts to this stack's own container names
  (`DASHBOARD_CONTAINERS` in `app.py`) even though the proxy itself can see
  every container on the host. `start-webui.ps1`/`.sh` now also detect and
  export `HOST_RAM_TOTAL_GB` (best effort) for the host-RAM gauge, since
  `model-status` can only see what's inside Docker Desktop's WSL2 VM
  otherwise.
- Dashboard: two more gauges, **GPU Memory (dedicated)** and **GPU Memory
  (shared)**, mirroring Windows Task Manager's GPU tab. Dedicated VRAM
  comes from `nvidia-smi --query-gpu=memory.used,memory.total`, queryable
  from inside the container. "Shared" GPU memory (pinned system RAM mapped
  for the GPU) is a Windows/WDDM concept with no Linux/nvidia-smi
  equivalent, so `start-webui.ps1`/`.sh` read it host-side via
  `Get-Counter '\GPU Adapter Memory(*)\Shared Usage'` (best effort, and
  Windows-only -- skipped on native Linux) and pass it in as
  `HOST_GPU_SHARED_USED_GB`/`HOST_GPU_SHARED_TOTAL_GB`. There's no "Shared
  Limit" perf counter on most systems, so the total is an *estimate* (half
  of `HOST_RAM_TOTAL_GB`, Windows' default shared-memory pool policy) and
  is labeled "(est.)" in the UI rather than presented as authoritative.

### Fixed (moe_infinity library)

These two are fixes to `moe_infinity/` itself (not the Docker/WebUI layer),
found while debugging why `allenai/OLMoE-1B-7B-0924-Instruct` never loaded
in this stack even though it's listed as a supported architecture:

- `moe_infinity/entrypoints/openai/api_server_v2.py`, `_initialize_model()`
  had no `except` clause, and is scheduled as a fire-and-forget
  `asyncio.create_task` that nothing ever awaits or inspects. Any exception
  raised while constructing the model (architecture errors, CUDA
  allocation failures, etc.) was silently dropped: the process kept
  running, `/health` stayed on `"starting"` forever, and every request got
  a `503` indistinguishable from a genuine hang. Now the failure is logged
  (`logger.exception(...)`) and surfaced through `/health` as
  `{"status": "unhealthy", "reason": "<exception>"}`. This is what
  actually revealed both root causes below -- before this fix, both looked
  identical from the outside (an eternal, silent `"starting"`).
- `moe_infinity/utils/hf_config.py`, `parse_moe_param()` /
  `parse_expert_id()`: `olmoe` is registered in
  `moe_infinity/common/constants.py`'s `MODEL_MAPPING_NAMES` /
  `MODEL_MAPPING_TYPES` and has a working monkey-patch class
  (`SyncOlmoeMoEBlock` in `moe_infinity/models/olmoe.py`), but was missing
  from the `if/elif` chain in both functions, which raised
  `RuntimeError: Unsupported architecture olmoeforcausallm`. OLMoE's config
  fields (`num_experts`, `num_experts_per_tok`) and expert parameter
  naming (`layers.N.mlp.experts.M....`) are identical to the already-handled
  `qwen3` branch, so `"olmoe" in arch` was added to that same branch in
  both functions. The same registration gap (present in `constants.py`,
  missing in `hf_config.py`) still exists for `dbrx`, `jamba`, and `opt`.
  With this fix, `allenai/OLMoE-1B-7B-0924-Instruct` loads and serves
  correctly (confirmed via `/v1/chat/completions`), though a separate,
  unrelated bug remains: the fused MoE CUDA kernel
  (`extensions/kernel/fused_moe_mlp.cu:123`,
  `fused_moe_ffn_into`/`MoEMLP::ForwardHelper`) throws `hidden dim mismatch`
  for OLMoE's expert shape (`intermediate_size=1024` vs
  `hidden_size=2048`), producing degenerate output and eventually crashing
  the process -- not fixed here, needs a native-kernel-level look.
- `moe_infinity/runtime/model_offload.py`, `setup_archer_hooks()`: found
  while hunting for a small, fast T0 model to validate the offload pipeline
  without the host-RAM pressure that crashes `DeepSeek-V2-Lite-Chat` (see
  Notes). Tiny test checkpoints commonly tie the LM head to the input
  embeddings to save space (`tie_word_embeddings: true`) -- when
  `lm_head.weight` has no separate tensor id for that reason, the code
  unconditionally treated it as "the NLLB MoE case" and reached for
  `model.model.encoder.embed_tokens` / `.decoder.embed_tokens`, which don't
  exist on any decoder-only architecture (`MixtralModel`, `Qwen3MoeModel`,
  etc.) -- `AttributeError: 'MixtralModel' object has no attribute
  'encoder'`. Now it checks whether the model actually has encoder/decoder
  submodules (true only for NLLB-MoE) and falls back to the single
  `model.model.embed_tokens` decoder-only models actually have. Confirmed
  fix against `vprovorg/tiny-random-Mixtral-8x7B-v0.1` (tiny, bfloat16,
  tied embeddings) -- loads, serves, and returns completions via both
  `/v1/completions` and `/v1/chat/completions` (the latter needed a
  `chat_template` patched into the cached tokenizer config too, since the
  checkpoint doesn't ship one -- a local cache edit, not a code fix).
  Separately confirmed the fused MoE CUDA kernel is BF16-only
  (`fused_moe_ffn_into: BF16 only`, `extensions/kernel/fused_moe_mlp.cu`) --
  an fp16 tiny checkpoint (`yujiepan/mixtral-tiny-random`) fails there
  regardless of this fix, which is why the bf16 `vprovorg` one was picked
  as this stack's default T0 model instead (see `docker-compose.webui.yml`,
  `.env.example`, `model-status/app.py`'s `KNOWN_MODELS`).

### Changed

- `docker/Dockerfile.serve` skips building `flash-attn` from source by
  default (`INSTALL_FLASH_ATTN` build arg, default `false`). flash-attn is a
  pure speed optimization with an automatic eager-attention fallback
  (`moe_infinity/runtime/model_offload.py`), but building it from source with
  no prebuilt wheel available was by far the slowest, most RAM/disk-hungry
  step of the build -- it was observed driving host RAM and disk I/O to
  100%, including a Docker Desktop WSL2 backend crash, on a 32 GB RAM / 24
  logical core host. `docker-compose.webui.yml` and the README document how
  to opt back in via `INSTALL_FLASH_ATTN=true`.
- Fixed: the first pass at excluding flash-attn only filtered the initial
  `pip install -r requirements.txt` layer. `setup.py` independently rebuilds
  `install_requires` from `requirements.txt` at editable-install time
  (`fetch_requirements("requirements.txt")`), and `COPY . .` re-copies the
  original, unfiltered file over the earlier filtered one -- so the `pip
  install -e .` step was still pulling in and building flash-attn from
  source regardless. `docker/Dockerfile.serve` now strips flash-attn from
  `requirements.txt` a second time, in-place, after `COPY . .`.
- [`check-system-requirements.ps1`](check-system-requirements.ps1): the disk
  check only ever looked at free space on `C:`, but Docker Desktop's WSL2
  backend stores images, volumes, and any named-volume model cache inside a
  distro's own virtual disk (`ext4.vhdx`), whose location is independent of
  the Windows install drive -- easy to move via Docker Desktop's Settings >
  Resources > Advanced, or `wsl --manage <Distro> --move`, and on this
  session's own dev machine the `docker-desktop`/`docker-desktop-data`
  distros do in fact live on `D:`, not `C:`. The script now also resolves
  every registered WSL distro's real location from
  `HKCU:\...\Lxss\<GUID>\BasePath` and checks free space on whichever
  drive(s) actually host them, alongside (not instead of) the existing `C:`
  check.
- `model-status`: merged the "Locally cached models" and "Look up a model"
  sections into a single **Models** table (every README-listed model plus
  whatever's already cached, with an "Add" field for anything else) showing
  download progress as a segmented meter, the disk space still needed for
  each (final size minus what's already downloaded, checked against current
  free disk and shown as a green/red dot), and update status with both
  commit ids. VRAM moved out of the per-model compatibility check into the
  System row's GPU stat tile, alongside a new CPU stat tile (core count +
  load, via `psutil.cpu_count()`/`cpu_percent()` -- previously absent from
  the dashboard entirely). The whole page was restyled as a dark,
  panel-based dashboard in the spirit of Grafana. `/api/check-model`,
  `/api/local-models`, and `/api/known-models` were folded into one
  `/api/models` endpoint (HuggingFace Hub lookups for all known + cached
  repos run in parallel via `ThreadPoolExecutor` rather than one-at-a-time
  on demand).

### Notes

- Requires an NVIDIA GPU with the NVIDIA Container Toolkit on the host.
- Distinct in purpose from `docker/Dockerfile` and `docker/Dockerfile.benchmark`,
  which build/test and benchmark images respectively and are not meant for
  interactive model use.
- `deepseek-ai/DeepSeek-V2-Lite-Chat` (~30GB) needs more RAM headroom than
  it looks like it should on a 32GB-RAM host: with WSL2 given 21.5GB, its
  offload-construction phase peaks around 94-95% memory and the topology
  setup's pinned-memory allocation (`cudaHostAlloc`, `model_offload.py:943`)
  fails at that pressure -- now surfaced as a clean `/health` error thanks
  to the `_initialize_model` exception-handling fix above, rather than an
  endless silent `"starting"`. `allenai/OLMoE-1B-7B-0924-Instruct` (~13GB)
  loads comfortably in the same environment (peaks ~93% only briefly,
  settles under 65%).
- On `cudaHostAlloc failed`: `self.archer_engine.set_topology(topo)`
  (`model_offload.py:943`) only builds a tensor-ID/stage index
  (`ArcherPrefetchHandle::SetTopology` -> `ArcherTopologyHandle::
  InitializeTopology`, `core/prefetch/archer_prefetch_handle.cpp:348`,
  `core/model/model_topology.cpp:507`) -- it doesn't itself move or pin the
  full ~30GB of weights, so the pinned-memory request that fails there is
  much smaller than "the whole offload". The pinned allocator behind it
  (`c10::HostCachingAllocator`, `core/memory/host_caching_allocator.cpp`)
  is a generic, reusable pool serviced by many separately-sized
  `cudaHostAlloc` calls over the process's life, **not** one allocation
  sized to the model -- and critically, its `free()` never actually
  releases pages back to CUDA/the OS (explicit comment in the source: "we
  are not really freeing the memory"), it only returns them to an internal
  reuse pool. So pinned-memory pressure is monotonically increasing within
  a single server process; freeing *host* RAM before starting a load still
  helps (pinned pages are carved from the same physical pool, and the
  allocation needs contiguous, lockable pages -- fails under pressure even
  when nominal "free" RAM looks nonzero), but restarting the container is
  what actually resets pinned usage back to zero.
