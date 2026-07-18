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

### Notes

- Requires an NVIDIA GPU with the NVIDIA Container Toolkit on the host.
- Distinct in purpose from `docker/Dockerfile` and `docker/Dockerfile.benchmark`,
  which build/test and benchmark images respectively and are not meant for
  interactive model use.
