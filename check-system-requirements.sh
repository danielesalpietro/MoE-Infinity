#!/usr/bin/env bash
# Pre-flight check: verifies the host meets the requirements to build and
# run the MoE-Infinity + Open WebUI stack (docker-compose.webui.yml) before
# you spend time on a build that will run out of RAM/VRAM/disk partway
# through. Run this before ./start-webui.sh.
#
# Usage: ./check-system-requirements.sh
set -uo pipefail

# Thresholds (GB unless noted). MIN = will very likely fail or thrash;
# RECOMMENDED = comfortable for a small MoE model like
# deepseek-ai/DeepSeek-V2-Lite-Chat (~30GB download).
MIN_CPU_CORES=4
RECOMMENDED_CPU_CORES=8
MIN_RAM_GB=16
RECOMMENDED_RAM_GB=32
MIN_VRAM_GB=8
RECOMMENDED_VRAM_GB=16
MIN_DISK_FREE_GB=40
RECOMMENDED_DISK_FREE_GB=80

pass_count=0
warn_count=0
fail_count=0

pass() { echo "  [PASS] $1"; pass_count=$((pass_count + 1)); }
warn() { echo "  [WARN] $1"; warn_count=$((warn_count + 1)); }
fail() { echo "  [FAIL] $1"; fail_count=$((fail_count + 1)); }

echo "== Docker =="
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    pass "Docker is installed and the daemon is reachable"
  else
    fail "Docker is installed but the daemon is not reachable (is Docker Desktop/dockerd running?)"
  fi
  if docker compose version >/dev/null 2>&1; then
    pass "Docker Compose v2 is available ($(docker compose version --short 2>/dev/null))"
  else
    fail "Docker Compose v2 plugin not found (docker compose ...)"
  fi
else
  fail "Docker is not installed or not on PATH"
fi

echo
echo "== CPU =="
cpu_cores=""
if command -v nproc >/dev/null 2>&1; then
  cpu_cores=$(nproc)
elif [ -f /proc/cpuinfo ]; then
  cpu_cores=$(grep -c ^processor /proc/cpuinfo)
fi
if [ -n "$cpu_cores" ]; then
  if [ "$cpu_cores" -ge "$RECOMMENDED_CPU_CORES" ]; then
    pass "$cpu_cores logical CPU cores (recommended: ${RECOMMENDED_CPU_CORES}+)"
  elif [ "$cpu_cores" -ge "$MIN_CPU_CORES" ]; then
    warn "$cpu_cores logical CPU cores (minimum: $MIN_CPU_CORES, recommended: ${RECOMMENDED_CPU_CORES}+)"
  else
    fail "$cpu_cores logical CPU cores (below minimum: $MIN_CPU_CORES)"
  fi
else
  warn "Could not determine CPU core count"
fi

echo
echo "== RAM =="
ram_gb=""
if command -v free >/dev/null 2>&1; then
  ram_gb=$(free -g | awk '/^Mem:/{print $2}')
fi
if [ -n "$ram_gb" ]; then
  if [ "$ram_gb" -ge "$RECOMMENDED_RAM_GB" ]; then
    pass "${ram_gb}GB total RAM (recommended: ${RECOMMENDED_RAM_GB}GB+)"
  elif [ "$ram_gb" -ge "$MIN_RAM_GB" ]; then
    warn "${ram_gb}GB total RAM (minimum: ${MIN_RAM_GB}GB, recommended: ${RECOMMENDED_RAM_GB}GB+). MoE-Infinity offloads experts to host RAM as an intermediate tier -- tight RAM here causes heavy swapping."
  else
    fail "${ram_gb}GB total RAM (below minimum: ${MIN_RAM_GB}GB)"
  fi
else
  warn "Could not determine total RAM (no 'free' command -- on Windows, run check-system-requirements.ps1 instead)"
fi

echo
echo "== GPU / VRAM =="
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_line=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$gpu_line" ]; then
    gpu_name=$(echo "$gpu_line" | cut -d',' -f1 | sed 's/^ *//;s/ *$//')
    vram_mb=$(echo "$gpu_line" | cut -d',' -f2 | sed 's/^ *//;s/ *$//')
    vram_gb=$((vram_mb / 1024))
    if [ "$vram_gb" -ge "$RECOMMENDED_VRAM_GB" ]; then
      pass "$gpu_name, ${vram_gb}GB VRAM (recommended: ${RECOMMENDED_VRAM_GB}GB+)"
    elif [ "$vram_gb" -ge "$MIN_VRAM_GB" ]; then
      warn "$gpu_name, ${vram_gb}GB VRAM (minimum: ${MIN_VRAM_GB}GB). Lower MOE_DEVICE_MEMORY_RATIO if you hit OOM."
    else
      fail "$gpu_name, ${vram_gb}GB VRAM (below minimum: ${MIN_VRAM_GB}GB)"
    fi
  else
    fail "nvidia-smi found but returned no GPU -- driver or GPU access problem"
  fi
else
  fail "nvidia-smi not found -- MoE-Infinity requires an NVIDIA GPU with the NVIDIA Container Toolkit installed and reachable from this shell"
fi

echo
echo "== Disk space =="
disk_free_gb=""
if command -v df >/dev/null 2>&1; then
  disk_free_gb=$(df -BG . 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
fi
if [ -n "$disk_free_gb" ]; then
  if [ "$disk_free_gb" -ge "$RECOMMENDED_DISK_FREE_GB" ]; then
    pass "${disk_free_gb}GB free on this filesystem (recommended: ${RECOMMENDED_DISK_FREE_GB}GB+)"
  elif [ "$disk_free_gb" -ge "$MIN_DISK_FREE_GB" ]; then
    warn "${disk_free_gb}GB free (minimum: ${MIN_DISK_FREE_GB}GB). The model download (~30GB) plus the built image (~15-30GB) and offload storage can get tight."
  else
    fail "${disk_free_gb}GB free (below minimum: ${MIN_DISK_FREE_GB}GB)"
  fi
else
  warn "Could not determine free disk space"
fi

echo
echo "== Summary =="
echo "  $pass_count passed, $warn_count warning(s), $fail_count failure(s)"
if [ "$fail_count" -gt 0 ]; then
  echo "  Result: NOT READY -- resolve the [FAIL] items above before running ./start-webui.sh"
  exit 2
elif [ "$warn_count" -gt 0 ]; then
  echo "  Result: READY WITH WARNINGS -- ./start-webui.sh should work but may be slow or need lower MOE_* ratios"
  exit 1
else
  echo "  Result: READY"
  exit 0
fi
