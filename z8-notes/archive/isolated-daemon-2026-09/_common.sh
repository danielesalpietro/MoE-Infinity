#!/usr/bin/env bash
# Shared configuration and helpers for the MoE-Infinity daemon scripts.
# Sourced by the numbered scripts; not meant to be run directly.

# --- what we are building -------------------------------------------------
# A second Docker daemon, fully separate from the system one. The system
# daemon on this host is managed by vast.ai: its kaalia agent polls
# `docker images -q` on the default socket and enforces a ~46.6 GB image
# budget, deleting whatever overflows it. It deleted our 30+ GB serving
# image 39 seconds after the build tagged it.
#
# A daemon with its own socket and its own data-root is not in that list, so
# nothing it holds can be reaped. Putting the data-root on PMEM also keeps us
# out of the 465 GB filesystem kaalia measures for free space, so we do not
# push it into pruning its own images more aggressively either.

# Docker 29 stores images in containerd content store, not in dockerd data-root.
# A dockerd started without an explicit --containerd attaches to the system
# containerd at /run/containerd/containerd.sock, and then both daemons list the
# same images -- which is exactly what the isolation check caught. So this
# stack runs its OWN containerd, with its own root, state and socket.
CONTAINERD_SERVICE="containerd-moe"
CONTAINERD_UNIT="/etc/systemd/system/${CONTAINERD_SERVICE}.service"
CONTAINERD_SOCK="/run/containerd-moe/containerd.sock"
CONTAINERD_STATE="/run/containerd-moe"

SERVICE_NAME="docker-moe"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
CONFIG_DIR="/etc/docker-moe"
CONFIG_FILE="${CONFIG_DIR}/daemon.json"
CONTAINERD_CONFIG="${CONFIG_DIR}/containerd.toml"
DOCKER_SOCK="/run/docker-moe.sock"
export DOCKER_HOST="unix://${DOCKER_SOCK}"

# Two filesystems, two jobs.
#
# The image store goes on the big SATA disk, next to (but separate from) the
# system daemon's own data-root. Image layers are read once when a container
# starts; they gain nothing from persistent memory, and parking tens of GB of
# them on PMEM would spend the scarce resource on the workload that least
# needs it.
#
# Isolation from vast.ai does not depend on this choice. kaalia enumerates
# images with `docker images -q` against the *system socket*; our daemon is
# not on that socket, so its images are invisible to it wherever the layers
# physically live. Its prune budget is a fixed 10% of the total disk
# (46.6 GB), computed from images it can see -- not from free space -- so our
# layers sharing the filesystem does not make it prune harder either.
STORE_MOUNT="/mnt/wdc-docker"
STORE_ROOT="${STORE_MOUNT}/docker-moe"
DATA_ROOT="${STORE_ROOT}/data"
EXEC_ROOT="${STORE_ROOT}/exec"
CONTAINERD_ROOT="${STORE_ROOT}/containerd"

# The system daemon's root, for the "are we actually separate?" assertion.
SYSTEM_DATA_ROOT="/mnt/wdc-docker/docker"

# PMEM is reserved for the workload: HF cache, expert offload, chat database.
PMEM_MOUNT="/mnt/pmem_emh2"
WORKLOAD_ROOT="${PMEM_MOUNT}/moe-infinity"

# Bridge + subnets chosen to avoid the 172.17-172.26 range already used by
# docker0 and the northstream/caliper compose networks on this host.
BRIDGE_NAME="docker-moe0"
BRIDGE_CIDR="172.30.0.1/16"
POOL_BASE="172.31.0.0/16"

REAL_NVIDIA_RUNTIME="/usr/bin/nvidia-container-runtime"

# Minimum free space on the image-store filesystem, in GB. The serving image
# alone is tens of GB, and this disk is shared with vast.ai's renters -- leave
# room rather than being the reason their pulls start failing.
MIN_FREE_GB=80

# --- output ---------------------------------------------------------------
if [ -t 1 ]; then
  C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
  C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_OFF=""
fi

ok()   { printf '%s  OK  %s %s\n' "$C_OK"   "$C_OFF" "$*"; }
warn() { printf '%s WARN %s %s\n' "$C_WARN" "$C_OFF" "$*"; }
fail() { printf '%s FAIL %s %s\n' "$C_ERR"  "$C_OFF" "$*"; }
info() { printf '%s      %s\n' "$C_DIM" "$*$C_OFF"; }
step() { printf '\n== %s ==\n' "$*"; }

# --- predicates -----------------------------------------------------------
containerd_active()  { systemctl is-active --quiet "$CONTAINERD_SERVICE" 2>/dev/null; }
containerd_responds() { [ -S "$CONTAINERD_SOCK" ]; }

# Which containerd socket is our dockerd actually attached to? Read it back
# from the daemon rather than trusting what we wrote into the config.
dockerd_containerd_addr() {
  docker -H "unix://${DOCKER_SOCK}" info --format "{{.ContainerdCommit.Expected}}" >/dev/null 2>&1 || true
  local pid; pid="$(pgrep -f "dockerd --config-file ${CONFIG_FILE}" | head -1)"
  [ -n "$pid" ] || return 0
  # xargs -0, not tr: NUL bytes do not survive a command substitution.
  xargs -0 echo < "/proc/${pid}/cmdline" 2>/dev/null
}

daemon_unit_exists() { [ -f "$UNIT_PATH" ]; }
daemon_active()      { systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; }
daemon_responds()    { docker -H "unix://${DOCKER_SOCK}" info >/dev/null 2>&1; }

require_sudo() {
  if ! sudo -n true 2>/dev/null; then
    fail "passwordless sudo is not available for $(whoami); run this from a shell where it is"
    exit 1
  fi
}

# Dump the unit's journal. Called whenever a start attempt fails, so the
# reason is on screen instead of behind a "see journalctl" suggestion.
dump_journal() {
  local unit="${2:-$SERVICE_NAME}"
  echo
  printf '%s--- journalctl -u %s ---%s
' "$C_DIM" "$unit" "$C_OFF"
  sudo -n journalctl -u "$unit" -n "${1:-30}" --no-pager 2>/dev/null     | sed 's/^/    /'     || echo "    (journal unreadable)"
  printf '%s--- end of journal ---%s

' "$C_DIM" "$C_OFF"
}

# dockerd rejects -b and --bip together, so the subnet cannot be set in
# daemon.json alongside a custom bridge name. Creating the bridge ourselves
# with the address we want and passing only its name makes dockerd adopt the
# existing configuration. Setting "bip" without "bridge" is not an option: it
# would apply to the default bridge, docker0, which belongs to the system
# daemon.
ensure_bridge() {
  if ip link show "$BRIDGE_NAME" >/dev/null 2>&1; then
    if ip -4 addr show "$BRIDGE_NAME" | grep -q "${BRIDGE_CIDR%%/*}"; then
      return 0   # already exactly as we want it
    fi
    sudo -n ip addr add "$BRIDGE_CIDR" dev "$BRIDGE_NAME" 2>/dev/null || true
  else
    sudo -n ip link add name "$BRIDGE_NAME" type bridge || return 1
    sudo -n ip addr add "$BRIDGE_CIDR" dev "$BRIDGE_NAME" || return 1
  fi
  sudo -n ip link set "$BRIDGE_NAME" up || return 1
}

free_gb() {  # free_gb <mountpoint>
  df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}
store_free_gb() { free_gb "$STORE_MOUNT"; }
pmem_free_gb()  { free_gb "$PMEM_MOUNT"; }

# Refuse to rm -rf anything that is not unmistakably ours. These scripts run
# with sudo on a host shared with other people containers, so a variable gone
# empty or mis-edited must abort, not delete a filesystem.
assert_our_path() {
  local path="$1" label="${2:-path}"
  case "$path" in
    "")  fail "${label} is empty -- refusing to delete"; exit 1 ;;
    /|/mnt|/etc|/run|/var|/home)
         fail "${label} is ${path} -- refusing to delete"; exit 1 ;;
    /*)  ;;
    *)   fail "${label} is not absolute: ${path}"; exit 1 ;;
  esac
  case "$path" in
    *docker-moe*|*containerd-moe*) ;;
    *) fail "${label} (${path}) does not look like ours -- refusing"; exit 1 ;;
  esac
  if [ "$path" = "$SYSTEM_DATA_ROOT" ]; then
    fail "${label} IS the system daemon root -- refusing"; exit 1
  fi
  case "$SYSTEM_DATA_ROOT" in
    "$path"/*) fail "${label} contains the system daemon root -- refusing"; exit 1 ;;
  esac
  return 0
}
