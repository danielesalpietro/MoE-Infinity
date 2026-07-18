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

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import psutil

HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", "/root/.cache/huggingface"))
HF_HUB_DIR = HF_CACHE_DIR / "hub"
HF_TOKEN = os.environ.get("HF_TOKEN") or None

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
