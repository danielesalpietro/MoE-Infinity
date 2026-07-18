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
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
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


@app.get("/api/local-models")
def local_models() -> JSONResponse:
    if not HF_HUB_DIR.is_dir():
        return JSONResponse({"models": []})

    models = []
    for entry in sorted(HF_HUB_DIR.iterdir()):
        if entry.is_dir() and entry.name.startswith("models--"):
            models.append(_local_model_status(entry))

    return JSONResponse({"models": models})


@app.get("/api/known-models")
def known_models() -> JSONResponse:
    return JSONResponse({"models": KNOWN_MODELS})


@app.get("/api/system-resources")
def system_resources() -> JSONResponse:
    ram_total_gb = _bytes_to_gb(psutil.virtual_memory().total)

    disk_free_gb = None
    try:
        usage = shutil.disk_usage(HF_CACHE_DIR if HF_CACHE_DIR.exists() else "/")
        disk_free_gb = _bytes_to_gb(usage.free)
    except OSError:
        pass

    gpu_name = None
    vram_total_gb = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        first_line = result.stdout.strip().splitlines()[0]
        name_part, vram_mib_part = first_line.rsplit(",", 1)
        gpu_name = name_part.strip()
        vram_total_gb = round(int(vram_mib_part.strip()) / 1024, 1)
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        IndexError,
        ValueError,
    ):
        pass

    return JSONResponse(
        {
            "ram_total_gb": ram_total_gb,
            "disk_free_gb": disk_free_gb,
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total_gb,
        }
    )


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

    base_resources = json.loads(system_resources().body)

    return JSONResponse(
        {
            "docker_ram": docker_ram,
            "host_ram": host_ram,
            "disk_system": disk_system,
            "disk_volumes": volumes,
            "disk_volumes_total_gb": _bytes_to_gb(total_volume_bytes),
            "gpu_name": base_resources.get("gpu_name"),
            "vram_total_gb": base_resources.get("vram_total_gb"),
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


def _compatibility(model_size_gb: float, resources: dict[str, Any]) -> dict[str, Any]:
    # Heuristics derived empirically in this session: MoE-Infinity's
    # offload-construction phase peaks at roughly 0.6-0.7x the model's
    # on-disk weight size in host RAM, and needs comparable free disk
    # headroom on top of the download itself. Treat these as rough
    # guidance, not a guarantee -- actual peak usage depends on the model
    # architecture (number of experts, hidden size, etc).
    checks = []

    ram_total_gb = resources.get("ram_total_gb")
    if ram_total_gb is not None:
        recommended_ram = round(model_size_gb * 1.0, 1)
        min_ram = round(model_size_gb * 0.65, 1)
        if ram_total_gb >= recommended_ram:
            checks.append({"item": "RAM", "level": "pass",
                            "detail": f"{ram_total_gb}GB available (recommended {recommended_ram}GB+)"})
        elif ram_total_gb >= min_ram:
            checks.append({"item": "RAM", "level": "warn",
                            "detail": f"{ram_total_gb}GB available (minimum {min_ram}GB, recommended {recommended_ram}GB+) -- offload construction may be slow or swap"})
        else:
            checks.append({"item": "RAM", "level": "fail",
                            "detail": f"{ram_total_gb}GB available, below estimated minimum {min_ram}GB"})

    disk_free_gb = resources.get("disk_free_gb")
    if disk_free_gb is not None:
        needed_disk = round(model_size_gb * 1.3, 1)
        if disk_free_gb >= needed_disk:
            checks.append({"item": "Disk", "level": "pass",
                            "detail": f"{disk_free_gb}GB free (needs ~{needed_disk}GB for download + offload)"})
        else:
            checks.append({"item": "Disk", "level": "fail",
                            "detail": f"{disk_free_gb}GB free, below estimated ~{needed_disk}GB needed"})

    vram_total_gb = resources.get("vram_total_gb")
    if vram_total_gb is not None:
        recommended_vram = max(8.0, round(model_size_gb * 0.3, 1))
        if vram_total_gb >= recommended_vram:
            checks.append({"item": "VRAM", "level": "pass",
                            "detail": f"{vram_total_gb}GB VRAM (recommended {recommended_vram}GB+)"})
        elif vram_total_gb >= 8.0:
            checks.append({"item": "VRAM", "level": "warn",
                            "detail": f"{vram_total_gb}GB VRAM (recommended {recommended_vram}GB+); lower MOE_DEVICE_MEMORY_RATIO if you hit OOM"})
        else:
            checks.append({"item": "VRAM", "level": "fail",
                            "detail": f"{vram_total_gb}GB VRAM, below the 8GB floor MoE-Infinity needs"})
    else:
        checks.append({"item": "VRAM", "level": "warn", "detail": "no NVIDIA GPU detected from this container"})

    levels = [c["level"] for c in checks]
    overall = "fail" if "fail" in levels else ("warn" if "warn" in levels else "pass")
    return {"overall": overall, "checks": checks}


@app.get("/api/check-model")
def check_model(repo_id: str = Query(..., min_length=1)) -> JSONResponse:
    api = HfApi(token=HF_TOKEN)
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except HfHubHTTPError as exc:
        raise HTTPException(status_code=502, detail=f"HuggingFace Hub lookup failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 -- surface any lookup failure to the caller
        raise HTTPException(status_code=502, detail=f"HuggingFace Hub lookup failed: {exc}") from exc

    size_bytes = _hf_model_size_bytes(info)
    size_gb = _bytes_to_gb(size_bytes)
    resources = system_resources().body
    import json as _json
    resources_dict = _json.loads(resources)

    local_dirname = "models--" + repo_id.replace("/", "--")
    local_entry = None
    local_path = HF_HUB_DIR / local_dirname
    if local_path.is_dir():
        local_entry = _local_model_status(local_path)

    update_available = None
    if local_entry and local_entry.get("local_commit"):
        update_available = local_entry["local_commit"] != info.sha

    return JSONResponse(
        {
            "repo_id": repo_id,
            "remote_commit": info.sha,
            "size_bytes": size_bytes,
            "size_gb": size_gb,
            "gated": bool(getattr(info, "gated", False)),
            "local": local_entry,
            "update_available": update_available,
            "compatibility": _compatibility(size_gb, resources_dict),
            "start_command": f"MOE_MODEL={repo_id} ./start-webui.sh   (or  .\\start-webui.ps1 -Model {repo_id})",
        }
    )


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
