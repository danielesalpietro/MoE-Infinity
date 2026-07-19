"""Read-only model status page for the MoE-Infinity WebUI stack.

Shows which models are already cached locally (with size and whether the
download looks complete / whether HuggingFace Hub has a newer commit), and
lets you look up any HuggingFace repo id to see its download size and
whether this host's RAM/VRAM/disk look sufficient before you point
MOE_MODEL at it.

This service is intentionally read-only: it never starts a download or
touches the moe-infinity container. Switching models is still done via
`MOE_MODEL=<repo_id> ./start-webui.sh` (see README).
"""

import json
import os
import re
import shutil
import struct
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi
import psutil

HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", "/root/.cache/huggingface"))
HF_HUB_DIR = HF_CACHE_DIR / "hub"
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Optional: host total RAM in GB, passed in by start-webui.ps1/.sh (best
# effort -- psutil inside the container only sees what Docker Desktop's
# WSL2 VM was granted, not the physical host total).
HOST_RAM_TOTAL_GB = os.environ.get("HOST_RAM_TOTAL_GB")

# Read-only Docker API access via tecnativa/docker-socket-proxy (GET-only,
# no exec/start/stop/create -- see docker-compose.webui.yml). Optional: the
# dashboard degrades gracefully (container/log panels just report
# "unavailable") if this isn't set or isn't reachable.
DOCKER_PROXY_URL = (os.environ.get("DOCKER_PROXY_URL") or "").rstrip("/")

# Only these containers can be queried through the dashboard, even though
# the proxy technically has visibility into every container on the host --
# this is our stack's dashboard, not a general Docker inspector.
DASHBOARD_CONTAINERS = [
    "moe-infinity-server",
    "moe-infinity-open-webui",
    "moe-infinity-model-status",
    "moe-infinity-docker-proxy",
]

# Volumes mounted read-only for disk accounting (see docker-compose.webui.yml).
DASHBOARD_VOLUME_PATHS = {
    "hf_cache": HF_CACHE_DIR,
    "offload": Path(os.environ.get("OFFLOAD_DIR", "/mnt/offload")),
    "open_webui_data": Path(os.environ.get("OPEN_WEBUI_DATA_DIR", "/mnt/open_webui_data")),
}

# Weight file extensions counted toward a model's download size.
WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth", ".gguf")

# Example checkpoints from the README's "Supported Models" table -- a
# starting point for the lookup form, not an exhaustive list.
KNOWN_MODELS = [
    # Tiny randomly-initialized checkpoints (a few MB) -- T0 smoke-test
    # baselines that exercise the full offload/serve pipeline without the
    # RAM/VRAM pressure of a real model. Not for actual serving quality.
    # Must declare torch_dtype=bfloat16 in config.json -- the fused MoE CUDA
    # kernel (extensions/kernel/fused_moe_mlp.cu) hard-requires BF16 and
    # throws "fused_moe_ffn_into: BF16 only" for fp16/fp32 checkpoints
    # (confirmed against yujiepan/mixtral-tiny-random, which is fp16).
    {"repo_id": "vprovorg/tiny-random-Mixtral-8x7B-v0.1", "family": "Mixtral (tiny/test, bf16)"},
    {"repo_id": "yujiepan/qwen3-moe-tiny-random", "family": "Qwen3-MoE (tiny/test, bf16)"},
    {"repo_id": "deepseek-ai/DeepSeek-V2-Lite-Chat", "family": "DeepSeek-V2"},
    {"repo_id": "deepseek-ai/DeepSeek-V3", "family": "DeepSeek-V3"},
    {"repo_id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "family": "Mixtral"},
    {"repo_id": "Qwen/Qwen3-30B-A3B", "family": "Qwen3-MoE"},
    {"repo_id": "openai/gpt-oss-20b", "family": "GPT-OSS"},
    {"repo_id": "databricks/dbrx-instruct", "family": "DBRX"},
    {"repo_id": "allenai/OLMoE-1B-7B-0924-Instruct", "family": "OLMoE"},
    {"repo_id": "facebook/nllb-moe-54b", "family": "Meta NLLB-MoE"},
]

app = FastAPI(title="MoE-Infinity Model Status")


def _bytes_to_gb(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 2)


def _repo_id_from_cache_dirname(dirname: str) -> Optional[str]:
    # HF cache layout: models--<org>--<name> (or models--<name> if no org)
    match = re.match(r"^models--(.+)$", dirname)
    if not match:
        return None
    return match.group(1).replace("--", "/", 1)


def _local_model_status(model_dir: Path) -> dict[str, Any]:
    repo_id = _repo_id_from_cache_dirname(model_dir.name)
    blobs_dir = model_dir / "blobs"
    snapshots_dir = model_dir / "snapshots"
    refs_dir = model_dir / "refs"

    size_bytes = 0
    if blobs_dir.is_dir():
        for entry in blobs_dir.iterdir():
            if entry.is_file():
                size_bytes += entry.stat().st_size

    incomplete_files = list(blobs_dir.glob("*.incomplete")) if blobs_dir.is_dir() else []

    local_commit = None
    main_ref = refs_dir / "main"
    if main_ref.is_file():
        local_commit = main_ref.read_text().strip()

    snapshot_complete = False
    if local_commit and snapshots_dir.is_dir():
        snapshot_path = snapshots_dir / local_commit
        if snapshot_path.is_dir():
            broken_links = [
                f for f in snapshot_path.rglob("*")
                if f.is_symlink() and not f.resolve().exists()
            ]
            snapshot_complete = len(broken_links) == 0

    return {
        "repo_id": repo_id,
        "size_bytes": size_bytes,
        "size_gb": _bytes_to_gb(size_bytes),
        "local_commit": local_commit,
        "complete": bool(local_commit) and snapshot_complete and not incomplete_files,
        "incomplete_file_count": len(incomplete_files),
    }


def _disk_free_gb() -> Optional[float]:
    try:
        usage = shutil.disk_usage(HF_CACHE_DIR if HF_CACHE_DIR.exists() else "/")
        return _bytes_to_gb(usage.free)
    except OSError:
        return None


def _gpu_info() -> dict[str, Any]:
    """Dedicated GPU memory via nvidia-smi -- real, always available on any
    host with GPU passthrough working. There is no Linux/nvidia-smi
    equivalent for Windows' "shared GPU memory" (WDDM concept, pinned
    system RAM mapped for the GPU) -- see HOST_GPU_SHARED_*_GB, passed in
    from the host by start-webui.ps1/.sh, for that."""
    info: dict[str, Any] = {
        "gpu_name": None,
        "vram_total_gb": None,
        "vram_used_gb": None,
        "vram_used_pct": None,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        first_line = result.stdout.strip().splitlines()[0]
        name_part, total_part, used_part = first_line.split(",")
        info["gpu_name"] = name_part.strip()
        total_gb = round(int(total_part.strip()) / 1024, 1)
        used_gb = round(int(used_part.strip()) / 1024, 2)
        info["vram_total_gb"] = total_gb
        info["vram_used_gb"] = used_gb
        if total_gb:
            info["vram_used_pct"] = round(used_gb / total_gb * 100, 1)
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        IndexError,
        ValueError,
    ):
        pass
    return info


def _host_gpu_shared() -> Optional[dict[str, Any]]:
    used = os.environ.get("HOST_GPU_SHARED_USED_GB")
    total = os.environ.get("HOST_GPU_SHARED_TOTAL_GB")
    if not used or not total:
        return None
    try:
        used_gb = float(used)
        total_gb = float(total)
    except ValueError:
        return None
    return {
        "used_gb": used_gb,
        "total_gb": total_gb,
        "used_pct": round(used_gb / total_gb * 100, 1) if total_gb else None,
    }


def _dir_size_bytes(path: Path) -> Optional[int]:
    if not path.is_dir():
        return None
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


@app.get("/api/dashboard/resources")
def dashboard_resources() -> JSONResponse:
    mem = psutil.virtual_memory()
    docker_ram = {
        "total_gb": _bytes_to_gb(mem.total),
        "used_gb": _bytes_to_gb(mem.used),
        "used_pct": mem.percent,
    }
    cpu = {
        "count": psutil.cpu_count(logical=True),
        "used_pct": psutil.cpu_percent(interval=0.1),
    }

    host_ram = None
    if HOST_RAM_TOTAL_GB:
        try:
            host_total = float(HOST_RAM_TOTAL_GB)
            host_ram = {
                "total_gb": host_total,
                # Best-effort: we don't have visibility into what's used
                # outside the Docker Desktop VM, so this only shows the
                # portion Docker itself is using against the true host total.
                "docker_used_gb": docker_ram["used_gb"],
                "docker_used_pct": round(docker_ram["used_gb"] / host_total * 100, 1) if host_total else None,
            }
        except ValueError:
            host_ram = None

    disk_system = None
    try:
        usage = shutil.disk_usage("/")
        disk_system = {
            "total_gb": _bytes_to_gb(usage.total),
            "used_gb": _bytes_to_gb(usage.used),
            "free_gb": _bytes_to_gb(usage.free),
            "used_pct": round(usage.used / usage.total * 100, 1) if usage.total else None,
        }
    except OSError:
        pass

    volumes = {}
    total_volume_bytes = 0
    for label, path in DASHBOARD_VOLUME_PATHS.items():
        size = _dir_size_bytes(path)
        volumes[label] = _bytes_to_gb(size) if size is not None else None
        if size is not None:
            total_volume_bytes += size

    gpu = _gpu_info()
    gpu_dedicated = None
    if gpu["vram_total_gb"] is not None:
        gpu_dedicated = {
            "used_gb": gpu["vram_used_gb"],
            "total_gb": gpu["vram_total_gb"],
            "used_pct": gpu["vram_used_pct"],
        }
    gpu_shared = _host_gpu_shared()

    return JSONResponse(
        {
            "cpu": cpu,
            "docker_ram": docker_ram,
            "host_ram": host_ram,
            "disk_system": disk_system,
            "disk_volumes": volumes,
            "disk_volumes_total_gb": _bytes_to_gb(total_volume_bytes),
            "gpu_name": gpu["gpu_name"],
            "vram_total_gb": gpu["vram_total_gb"],
            "gpu_dedicated": gpu_dedicated,
            "gpu_shared": gpu_shared,
        }
    )


def _docker_api_get(path: str, timeout: float = 5.0) -> Any:
    if not DOCKER_PROXY_URL:
        return None
    try:
        with urllib.request.urlopen(f"{DOCKER_PROXY_URL}{path}", timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _docker_api_get_raw(path: str, timeout: float = 5.0) -> Optional[bytes]:
    if not DOCKER_PROXY_URL:
        return None
    try:
        with urllib.request.urlopen(f"{DOCKER_PROXY_URL}{path}", timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError):
        return None


def _cpu_percent(stats: dict[str, Any]) -> Optional[float]:
    try:
        cpu_delta = (
            stats["cpu_stats"]["cpu_usage"]["total_usage"]
            - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        )
        system_delta = (
            stats["cpu_stats"]["system_cpu_usage"]
            - stats["precpu_stats"]["system_cpu_usage"]
        )
        num_cpus = stats["cpu_stats"].get("online_cpus") or len(
            stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1])
        )
        if system_delta > 0 and cpu_delta >= 0:
            return round((cpu_delta / system_delta) * num_cpus * 100, 1)
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    return None


@app.get("/api/dashboard/containers")
def dashboard_containers() -> JSONResponse:
    if not DOCKER_PROXY_URL:
        return JSONResponse({"available": False, "containers": []})

    all_containers = _docker_api_get("/containers/json?all=1")
    if all_containers is None:
        return JSONResponse({"available": False, "containers": []})

    by_name = {}
    for c in all_containers:
        names = [n.lstrip("/") for n in c.get("Names", [])]
        for name in names:
            if name in DASHBOARD_CONTAINERS:
                by_name[name] = c

    containers = []
    for name in DASHBOARD_CONTAINERS:
        c = by_name.get(name)
        if c is None:
            containers.append({"name": name, "status": "not found", "state": None,
                                "cpu_pct": None, "mem_used_gb": None, "mem_pct": None})
            continue

        state = c.get("State")
        cpu_pct = None
        mem_used_gb = None
        mem_pct = None
        if state == "running":
            stats = _docker_api_get(f"/containers/{c['Id']}/stats?stream=false", timeout=8.0)
            if stats:
                cpu_pct = _cpu_percent(stats)
                mem_usage = stats.get("memory_stats", {}).get("usage")
                mem_limit = stats.get("memory_stats", {}).get("limit")
                if mem_usage is not None:
                    mem_used_gb = _bytes_to_gb(mem_usage)
                if mem_usage and mem_limit:
                    mem_pct = round(mem_usage / mem_limit * 100, 1)

        containers.append(
            {
                "name": name,
                "status": c.get("Status"),
                "state": state,
                "cpu_pct": cpu_pct,
                "mem_used_gb": mem_used_gb,
                "mem_pct": mem_pct,
            }
        )

    return JSONResponse({"available": True, "containers": containers})


def _demux_docker_logs(raw: bytes) -> str:
    # Non-TTY containers' log stream is framed: 1 byte stream type, 3 bytes
    # padding, 4 bytes big-endian payload length, then that many payload
    # bytes, repeated. https://docs.docker.com/engine/api/v1.41/#tag/Container/operation/ContainerAttach
    lines = []
    offset = 0
    while offset + 8 <= len(raw):
        length = struct.unpack(">I", raw[offset + 4:offset + 8])[0]
        start = offset + 8
        end = start + length
        lines.append(raw[start:end].decode("utf-8", errors="replace"))
        offset = end
    if not lines and raw:
        # Fallback: not framed (can happen for proxies that already strip
        # headers) -- return as-is.
        return raw.decode("utf-8", errors="replace")
    return "".join(lines)


@app.get("/api/dashboard/logs", response_class=PlainTextResponse)
def dashboard_logs(
    container: str = Query(..., description="Container name"),
    tail: int = Query(200, ge=1, le=2000),
) -> str:
    if container not in DASHBOARD_CONTAINERS:
        raise HTTPException(status_code=400, detail="Unknown container")
    if not DOCKER_PROXY_URL:
        return "Docker API proxy not configured -- logs unavailable."

    raw = _docker_api_get_raw(
        f"/containers/{container}/logs?stdout=1&stderr=1&tail={tail}&timestamps=1"
    )
    if raw is None:
        return f"Could not reach Docker API proxy for {container}'s logs."
    return _demux_docker_logs(raw) or "(no log output yet)"


def _hf_model_size_bytes(info: Any) -> int:
    total = 0
    for sibling in info.siblings or []:
        filename = getattr(sibling, "rfilename", "") or ""
        size = getattr(sibling, "size", None)
        if size and filename.endswith(WEIGHT_EXTENSIONS):
            total += size
    return total


def _remote_model_info(repo_id: str) -> dict[str, Any]:
    api = HfApi(token=HF_TOKEN)
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001 -- surface any lookup failure per-row instead of failing the whole dashboard
        return {"repo_id": repo_id, "error": str(exc)}

    size_bytes = _hf_model_size_bytes(info)
    return {
        "repo_id": repo_id,
        "size_bytes": size_bytes,
        "size_gb": _bytes_to_gb(size_bytes),
        "commit": info.sha,
        "gated": bool(getattr(info, "gated", False)),
    }


@app.get("/api/models")
def all_models(extra: str = Query("", description="Comma-separated extra repo_ids to include")) -> JSONResponse:
    local_by_repo: dict[str, dict[str, Any]] = {}
    if HF_HUB_DIR.is_dir():
        for entry in sorted(HF_HUB_DIR.iterdir()):
            if entry.is_dir() and entry.name.startswith("models--"):
                status = _local_model_status(entry)
                if status["repo_id"]:
                    local_by_repo[status["repo_id"]] = status

    extra_ids = [r.strip() for r in extra.split(",") if r.strip()]
    repo_ids = sorted({*(m["repo_id"] for m in KNOWN_MODELS), *local_by_repo.keys(), *extra_ids})

    remote_by_repo: dict[str, dict[str, Any]] = {}
    if repo_ids:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_remote_model_info, repo_id) for repo_id in repo_ids]
            for fut in as_completed(futures):
                result = fut.result()
                remote_by_repo[result["repo_id"]] = result

    disk_free_gb = _disk_free_gb()

    models = []
    for repo_id in repo_ids:
        local = local_by_repo.get(repo_id)
        remote = remote_by_repo.get(repo_id, {})
        remote_size_gb = remote.get("size_gb")
        local_size_gb = local["size_gb"] if local else 0.0

        pct_downloaded = None
        if remote_size_gb:
            pct_downloaded = round(min(100.0, (local_size_gb / remote_size_gb) * 100), 1)

        delta_gb = None
        disk_ok = None
        if remote_size_gb is not None:
            delta_gb = round(max(0.0, remote_size_gb - local_size_gb), 2)
            if disk_free_gb is not None:
                disk_ok = disk_free_gb >= delta_gb

        update_available = None
        local_commit = local.get("local_commit") if local else None
        remote_commit = remote.get("commit")
        if local_commit and remote_commit:
            update_available = local_commit != remote_commit

        models.append(
            {
                "repo_id": repo_id,
                "downloaded": local is not None,
                "complete": bool(local["complete"]) if local else False,
                "local_size_gb": local_size_gb,
                "remote_size_gb": remote_size_gb,
                "pct_downloaded": pct_downloaded,
                "delta_gb": delta_gb,
                "disk_ok": disk_ok,
                "local_commit": local_commit,
                "remote_commit": remote_commit,
                "update_available": update_available,
                "gated": remote.get("gated", False),
                "lookup_error": remote.get("error"),
            }
        )

    return JSONResponse({"models": models, "disk_free_gb": disk_free_gb})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
