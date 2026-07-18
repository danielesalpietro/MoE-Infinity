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

### Notes

- Requires an NVIDIA GPU with the NVIDIA Container Toolkit on the host.
- Distinct in purpose from `docker/Dockerfile` and `docker/Dockerfile.benchmark`,
  which build/test and benchmark images respectively and are not meant for
  interactive model use.
