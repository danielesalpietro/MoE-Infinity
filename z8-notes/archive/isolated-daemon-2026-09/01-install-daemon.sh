#!/usr/bin/env bash
# Create and start the isolated Docker daemon for MoE-Infinity.
#
# Idempotent: if the daemon is already up this exits 0 without touching
# anything. Safe to re-run after a reboot or a partial run.
#
# It never modifies the system Docker daemon, its config, its containers or
# its images, and it never touches vast.ai.

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=_common.sh
source ./_common.sh

step "Preflight"

# 1. Already running? Only skip if it is running *correctly*.
#
# "Active and responding" is not sufficient. A daemon started before this
# script grew a dedicated containerd will be up and healthy while quietly
# sharing the system image store -- which defeats the entire point, since
# vast.ai would still see and prune our images. Check the configuration, not
# just the liveness.
if daemon_active && daemon_responds; then
  needs_reconfig=0

  if ! sudo -n grep -q "$CONTAINERD_SOCK" "$CONFIG_FILE" 2>/dev/null; then
    warn "running daemon is not configured with a dedicated containerd"
    needs_reconfig=1
  fi

  if ! containerd_active; then
    warn "${CONTAINERD_SERVICE} is not running"
    needs_reconfig=1
  fi

  # The decisive test: do we and the system daemon list the same images?
  ours="$(docker -H "unix://${DOCKER_SOCK}" images -q 2>/dev/null | sort -u)"
  theirs="$(docker -H unix:///var/run/docker.sock images -q 2>/dev/null | sort -u)"
  if [ -n "$ours" ] && [ "$ours" = "$theirs" ]; then
    warn "this daemon and the system daemon list an identical image set -- the store is shared"
    needs_reconfig=1
  fi

  if [ "$needs_reconfig" -eq 0 ]; then
    ok "${SERVICE_NAME} is already active, isolated and responding on ${DOCKER_SOCK}"
    info "data-root: $(docker -H "unix://${DOCKER_SOCK}" info --format "{{.DockerRootDir}}" 2>/dev/null)"
    info "nothing to do -- run ./02-check-daemon.sh for a full verification"
    exit 0
  fi

  warn "reconfiguring in place: stopping ${SERVICE_NAME} and rewriting its configuration"
  info "no images are lost -- the shared store belongs to the system daemon and stays untouched"
  sudo -n systemctl stop "$SERVICE_NAME" 2>/dev/null || true
  sudo -n systemctl reset-failed "$SERVICE_NAME" 2>/dev/null || true
fi

if daemon_active && ! daemon_responds; then
  fail "${SERVICE_NAME} is active but not answering on ${DOCKER_SOCK}"
  info "inspect it with: sudo journalctl -u ${SERVICE_NAME} -n 50 --no-pager"
  exit 1
fi

# A previous attempt may have left the unit in an auto-restart loop, which
# reports neither active nor cleanly stopped. Clear it before proceeding.
if daemon_unit_exists; then
  state="$(systemctl is-active "$SERVICE_NAME" 2>/dev/null || true)"
  if [ "$state" = "activating" ] || [ "$state" = "failed" ]; then
    warn "${SERVICE_NAME} is in state '${state}' from an earlier attempt -- clearing it"
    sudo -n systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    sudo -n systemctl reset-failed "$SERVICE_NAME" 2>/dev/null || true
    sudo -n systemctl stop "$CONTAINERD_SERVICE" 2>/dev/null || true
    sudo -n systemctl reset-failed "$CONTAINERD_SERVICE" 2>/dev/null || true
  else
    warn "unit ${UNIT_PATH} already exists but the service is stopped -- reusing it"
  fi
fi

require_sudo
ok "passwordless sudo available"

# 2. Image store on the SATA disk; refuse to land it on the root filesystem.
if ! mountpoint -q "$STORE_MOUNT"; then
  fail "${STORE_MOUNT} is not a mount point -- refusing to put the image store on the root filesystem"
  exit 1
fi
ok "${STORE_MOUNT} is mounted"

# Never nest inside the system daemon's own data-root.
case "$DATA_ROOT" in
  "$SYSTEM_DATA_ROOT"/*|"$SYSTEM_DATA_ROOT")
    fail "${DATA_ROOT} is inside the system daemon's root (${SYSTEM_DATA_ROOT})"
    exit 1 ;;
esac
ok "image store is beside, not inside, ${SYSTEM_DATA_ROOT}"

avail="$(store_free_gb)"
if [ -z "$avail" ] || [ "$avail" -lt "$MIN_FREE_GB" ]; then
  fail "only ${avail:-?} GB free on ${STORE_MOUNT}, want at least ${MIN_FREE_GB} GB"
  info "this disk is shared with vast.ai's renters -- do not be the one who fills it"
  exit 1
fi
ok "${avail} GB free on ${STORE_MOUNT} (shared with the system daemon)"

pmem_avail="$(pmem_free_gb)"
info "PMEM left for the workload (HF cache, offload): ${pmem_avail:-?} GB on ${PMEM_MOUNT}"

# 3. The real NVIDIA runtime, not vast.ai's kaalia_docker_shim.
if [ ! -x "$REAL_NVIDIA_RUNTIME" ]; then
  fail "${REAL_NVIDIA_RUNTIME} not found -- GPU containers would not start"
  exit 1
fi
ok "genuine NVIDIA container runtime present"

# 4. Do not collide with a subnet already in use by another stack.
for cidr in "$BRIDGE_CIDR" "$POOL_BASE"; do
  prefix="${cidr%%.*}.$(echo "$cidr" | cut -d. -f2)."
  # Exclude our own bridge. It carries BRIDGE_CIDR by design, left over from a
  # previous run of this script, and ensure_bridge below is idempotent --
  # treating it as a conflict would make the script refuse to run twice.
  conflict="$(ip -4 -br addr show 2>/dev/null               | grep " ${prefix}"               | grep -v "^${BRIDGE_NAME}[[:space:]]" || true)"
  if [ -n "$conflict" ]; then
    fail "subnet ${cidr} is already in use by another interface on this host"
    printf "       %s
" "$conflict"
    exit 1
  fi
done
if ip link show "$BRIDGE_NAME" >/dev/null 2>&1; then
  ok "subnets ${BRIDGE_CIDR} and ${POOL_BASE} are free (ignoring our own ${BRIDGE_NAME})"
else
  ok "subnets ${BRIDGE_CIDR} and ${POOL_BASE} are free"
fi

if ip link show "$BRIDGE_NAME" >/dev/null 2>&1; then
  info "bridge ${BRIDGE_NAME} already exists -- will be reused"
fi

# 5. Record the system daemon's state so we can prove we did not disturb it.
sys_images_before="$(docker -H unix:///var/run/docker.sock images -q 2>/dev/null | wc -l || echo '?')"
info "system daemon currently holds ${sys_images_before} images (recorded, not touched)"

step "Writing configuration"

sudo mkdir -p "$DATA_ROOT" "$EXEC_ROOT" "$CONTAINERD_ROOT" "$CONFIG_DIR"
ok "created ${STORE_ROOT} and ${CONFIG_DIR}"

# --- dedicated containerd --------------------------------------------------
# Without this, dockerd falls back to /run/containerd/containerd.sock -- the
# system instance -- and both daemons end up listing the same images, because
# in Docker 29 the image store belongs to containerd, not to dockerd.
sudo tee "$CONTAINERD_CONFIG" > /dev/null <<TOML
version = 4

# root, state and the gRPC address are passed as command-line flags in the
# unit below, NOT set here.
#
# containerd 2.3 moved the gRPC server into a plugin
# (io.containerd.server.v1.grpc) and ignores a top-level [grpc] table in a
# version-4 config. Setting the address here therefore did nothing, and the
# instance came up on containerd default -- /run/containerd/containerd.sock,
# the SYSTEM socket. It unlinked and rebound that path, and on shutdown
# removed it, leaving the host containerd without its socket file.
#
# Command-line flags take precedence over the config file and are not subject
# to schema changes between containerd versions, so they are used instead.
TOML
ok "wrote ${CONTAINERD_CONFIG}"

sudo tee "$CONTAINERD_UNIT" > /dev/null <<UNIT
[Unit]
Description=containerd runtime for the MoE-Infinity Docker daemon
Documentation=https://github.com/danielesalpietro/MoE-Infinity
After=network.target

[Service]
Type=notify
ExecStartPre=-/sbin/modprobe overlay
ExecStart=/usr/bin/containerd --config ${CONTAINERD_CONFIG} --address ${CONTAINERD_SOCK} --root ${CONTAINERD_ROOT}/root --state ${CONTAINERD_STATE}
Delegate=yes
KillMode=process
Restart=on-failure
RestartSec=5
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
TasksMax=infinity
OOMScoreAdjust=-999
RuntimeDirectory=containerd-moe
RuntimeDirectoryMode=0711

[Install]
WantedBy=multi-user.target
UNIT
ok "wrote ${CONTAINERD_UNIT}"

sudo tee "$CONFIG_FILE" > /dev/null <<JSON
{
  "containerd": "${CONTAINERD_SOCK}",
  "data-root": "${DATA_ROOT}",
  "exec-root": "${EXEC_ROOT}",
  "hosts": ["unix://${DOCKER_SOCK}"],
  "pidfile": "/run/docker-moe.pid",
  "group": "docker",
  "bridge": "${BRIDGE_NAME}",
  "default-address-pools": [{ "base": "${POOL_BASE}", "size": 24 }],
  "exec-opts": ["native.cgroupdriver=systemd"],
  "runtimes": {
    "nvidia": { "path": "${REAL_NVIDIA_RUNTIME}", "args": [] }
  }
}
JSON
ok "wrote ${CONFIG_FILE}"
info "note: no \"bip\" here on purpose -- dockerd refuses -b and --bip together,"
info "      so the subnet lives on the bridge itself (created below)"

if ensure_bridge; then
  ok "bridge ${BRIDGE_NAME} ready at ${BRIDGE_CIDR}"
  info "$(ip -4 -br addr show "$BRIDGE_NAME" 2>/dev/null)"
else
  fail "could not create or configure bridge ${BRIDGE_NAME}"
  exit 1
fi
info "note: no registry-mirrors -- the system daemon routes pulls through"
info "      docker*.vast.ai mirrors; this one goes straight to the registry"

sudo tee "$UNIT_PATH" > /dev/null <<UNIT
[Unit]
Description=Docker daemon for MoE-Infinity (isolated image store on PMEM)
Documentation=https://github.com/danielesalpietro/MoE-Infinity
After=network-online.target ${CONTAINERD_SERVICE}.service
Wants=network-online.target
Requires=${CONTAINERD_SERVICE}.service
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=notify
ExecStart=/usr/bin/dockerd --config-file ${CONFIG_FILE}
ExecReload=/bin/kill -s HUP \$MAINPID
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
LimitMEMLOCK=infinity
TasksMax=infinity
Delegate=yes
KillMode=process
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
ok "wrote ${UNIT_PATH}"
info "LimitMEMLOCK=infinity: expert offloading pins large host buffers"

step "Starting"

sudo systemctl daemon-reload

# Fingerprint the system containerd socket before touching anything. If our
# instance ends up bound to that path instead of its own, it silently takes
# over the host runtime -- so this is checked, not assumed.
SYS_CD_SOCK="/run/containerd/containerd.sock"
sys_sock_inode_before="$(sudo -n stat -c %i "$SYS_CD_SOCK" 2>/dev/null || echo none)"
info "system containerd socket inode before start: ${sys_sock_inode_before}"

if ! sudo systemctl start "$CONTAINERD_SERVICE"; then
  fail "could not start ${CONTAINERD_SERVICE}"
  dump_journal 30 "$CONTAINERD_SERVICE"
  exit 1
fi

for _ in $(seq 1 20); do
  containerd_responds && break
  sleep 1
done
if ! containerd_responds; then
  fail "${CONTAINERD_SERVICE} started but ${CONTAINERD_SOCK} never appeared"
  dump_journal 30 "$CONTAINERD_SERVICE"
  exit 1
fi
ok "${CONTAINERD_SERVICE} up, socket at ${CONTAINERD_SOCK}"

# Did we take over the host socket? Abort and undo if so.
sys_sock_inode_after="$(sudo -n stat -c %i "$SYS_CD_SOCK" 2>/dev/null || echo none)"
if [ "$sys_sock_inode_after" != "$sys_sock_inode_before" ]; then
  fail "our containerd rebound the SYSTEM socket ${SYS_CD_SOCK}"
  info "inode ${sys_sock_inode_before} -> ${sys_sock_inode_after}; this would hijack the host runtime"
  warn "stopping ${CONTAINERD_SERVICE} and restoring the system containerd"
  sudo -n systemctl stop "$CONTAINERD_SERVICE" 2>/dev/null || true
  sudo -n systemctl restart containerd.service 2>/dev/null || true
  sleep 2
  if sudo -n test -S "$SYS_CD_SOCK"; then
    ok "system containerd socket restored"
  else
    fail "system containerd socket is MISSING -- run: sudo systemctl restart containerd.service"
  fi
  exit 1
fi
ok "system containerd socket untouched (inode ${sys_sock_inode_after})"

# Deliberately not enabled: this starts only when you ask for it, never at boot.
if ! sudo systemctl start "$SERVICE_NAME"; then
  fail "systemctl start failed"
  dump_journal 30
  info "the daemon config is at ${CONFIG_FILE} -- fix it and re-run this script"
  exit 1
fi

for _ in $(seq 1 30); do
  daemon_responds && break
  sleep 1
done

if ! daemon_responds; then
  fail "service started but the daemon is not answering on ${DOCKER_SOCK} after 30s"
  dump_journal 30
  exit 1
fi

ok "${SERVICE_NAME} is up"
info "data-root: $(docker info --format '{{.DockerRootDir}}')"
info "runtimes:  $(docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}')"

sys_images_after="$(docker -H unix:///var/run/docker.sock images -q 2>/dev/null | wc -l || echo '?')"
if [ "$sys_images_before" = "$sys_images_after" ]; then
  ok "system daemon untouched (${sys_images_after} images, unchanged)"
else
  warn "system daemon image count moved ${sys_images_before} -> ${sys_images_after}"
  info "not necessarily us: vast.ai prunes on its own schedule"
fi

step "Next"
cat <<TXT

  The daemon is NOT enabled at boot. Start it again after a reboot with:
      sudo systemctl start ${SERVICE_NAME}

  Point your shell at it before using docker or compose:
      export DOCKER_HOST=unix://${DOCKER_SOCK}

  Verify everything, including GPU access:
      ~/moe-infinity/02-check-daemon.sh

  Then build the stack (about an hour, mostly CPU-bound compilation):
      cd ${PMEM_MOUNT}/MoE-Infinity
      export DOCKER_HOST=unix://${DOCKER_SOCK}
      docker compose -f docker-compose.webui.yml build

TXT
