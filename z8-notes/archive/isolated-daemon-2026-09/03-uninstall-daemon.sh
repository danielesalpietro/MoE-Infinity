#!/usr/bin/env bash
# Remove the isolated Docker daemon and put the host back exactly as it was.
#
# By default the image store on PMEM is KEPT, so re-running the installer
# gives you the images back without another hour of building. Pass --purge to
# delete it too.
#
#   ./03-uninstall-daemon.sh           # stop + remove unit and config, keep images
#   ./03-uninstall-daemon.sh --purge   # also delete the image store (asks first)
#   ./03-uninstall-daemon.sh --purge --yes
#
# Never touches the system Docker daemon, its containers, its images, its
# volumes, or vast.ai.

set -uo pipefail
cd "$(dirname "$0")"
# shellcheck source=_common.sh
source ./_common.sh

PURGE=0
ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --purge) PURGE=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --dry-run|-n) DRY_RUN=1 ;;
    *) fail "unknown argument: $arg"; exit 2 ;;
  esac
done

# Every mutating command goes through this, so --dry-run prints the exact
# command instead of running it. Verifying that this script leaves the system
# daemon alone should not require trusting the description of what it does.
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf "%s would run:%s %s
" "$C_DIM" "$C_OFF" "$*"
  else
    "$@"
  fi
}

if [ "$DRY_RUN" -eq 1 ]; then
  printf "
%s  DRY RUN -- nothing is changed.%s
" "$C_WARN" "$C_OFF"
  printf "  Lines marked \"would run\" are the exact commands. The OK lines after
"
  printf "  them describe what WOULD follow, not what happened.
"
fi

step "What will be removed"

echo "  service   ${SERVICE_NAME}"
echo "  service   ${CONTAINERD_SERVICE}"
echo "  unit      ${UNIT_PATH}"
echo "  unit      ${CONTAINERD_UNIT}"
echo "  config    ${CONFIG_DIR}"
if [ "$PURGE" -eq 1 ]; then
  size="$(sudo du -sh "$STORE_ROOT" 2>/dev/null | cut -f1)"
  echo "  store     ${STORE_ROOT}  ${size:-(absent)}   <-- DELETED (--purge)"
else
  echo "  store     ${STORE_ROOT}   <-- kept, images and all (pass --purge to delete)"
fi
echo
echo "  NOT touched: the system Docker daemon, its containers/images/volumes,"
echo "               vast.ai, and ${PMEM_MOUNT}/moe-infinity (HF cache, offload)."
echo

# Warn about anything still running in this daemon before we pull it down.
if daemon_responds; then
  running="$(docker -H "unix://${DOCKER_SOCK}" ps -q 2>/dev/null | wc -l)"
  if [ "$running" -gt 0 ]; then
    warn "${running} container(s) still running in this daemon:"
    docker -H "unix://${DOCKER_SOCK}" ps --format "         {{.Names}}  {{.Image}}  {{.Status}}" 2>/dev/null
    echo
  fi
fi

if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "Proceed? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *) info "aborted, nothing changed"; exit 0 ;;
  esac
fi

require_sudo

step "Stopping"

# dockerd first, then the containerd it depends on.
if daemon_active; then
  run sudo systemctl stop "$SERVICE_NAME" && ok "stopped ${SERVICE_NAME}"
else
  info "${SERVICE_NAME} was not running"
fi

if containerd_active; then
  run sudo systemctl stop "$CONTAINERD_SERVICE" && ok "stopped ${CONTAINERD_SERVICE}"
else
  info "${CONTAINERD_SERVICE} was not running"
fi
sudo -n systemctl reset-failed "$CONTAINERD_SERVICE" 2>/dev/null || true

if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
  sudo systemctl disable "$SERVICE_NAME" >/dev/null 2>&1 && ok "disabled ${SERVICE_NAME}"
fi

step "Removing"

for u in "$UNIT_PATH" "$CONTAINERD_UNIT"; do
  if [ -f "$u" ]; then
    run sudo rm -f "$u" && ok "removed ${u}"
  fi
done

sudo systemctl daemon-reload
sudo systemctl reset-failed "$SERVICE_NAME" 2>/dev/null || true
ok "systemd reloaded"

if [ -d "$CONFIG_DIR" ]; then
  assert_our_path "$CONFIG_DIR" "CONFIG_DIR"
  run sudo rm -rf "$CONFIG_DIR" && ok "removed ${CONFIG_DIR}"
fi

run sudo rm -f "$DOCKER_SOCK" /run/docker-moe.pid 2>/dev/null || true
assert_our_path "$CONTAINERD_STATE" "CONTAINERD_STATE"
run sudo rm -rf "$CONTAINERD_STATE" 2>/dev/null || true

# The bridge is left behind by dockerd; remove it only if it is down and ours.
# Only ever an interface we named. Never docker0, never an empty var.
case "$BRIDGE_NAME" in
  docker-moe*) ;;
  *) fail "BRIDGE_NAME is [${BRIDGE_NAME}] -- not ours, refusing"; exit 1 ;;
esac
if ip link show "$BRIDGE_NAME" >/dev/null 2>&1; then
  run sudo ip link set "$BRIDGE_NAME" down 2>/dev/null || true
  run sudo ip link delete "$BRIDGE_NAME" 2>/dev/null && ok "removed bridge ${BRIDGE_NAME}" \
    || warn "could not remove bridge ${BRIDGE_NAME} -- harmless, it is unused"
fi

if [ "$PURGE" -eq 1 ]; then
  if [ -d "$STORE_ROOT" ]; then
    assert_our_path "$STORE_ROOT" "STORE_ROOT"
    run sudo rm -rf "$STORE_ROOT" && ok "deleted image store ${STORE_ROOT}"
  fi
else
  if [ -d "$STORE_ROOT" ]; then
    info "image store kept at ${STORE_ROOT} ($(sudo du -sh "$STORE_ROOT" 2>/dev/null | cut -f1))"
    info "re-running ./01-install-daemon.sh will pick it up as-is"
  fi
fi

# The docker context is user config, not root config, so it is removed here
# rather than left pointing at a socket that no longer exists.
if docker context inspect moe >/dev/null 2>&1; then
  run docker context rm moe >/dev/null 2>&1 && ok "removed docker context 'moe'"
fi

step "System daemon still healthy?"

# Check the SOCKET, not just the API. When our containerd was misconfigured it
# unlinked /run/containerd/containerd.sock on shutdown; the system dockerd kept
# answering on an already-established connection, so an API-only check reported
# everything fine while every new client was broken.
SYS_CD_SOCK="/run/containerd/containerd.sock"
if sudo -n test -S "$SYS_CD_SOCK" 2>/dev/null; then
  ok "system containerd socket present (${SYS_CD_SOCK})"
else
  fail "system containerd socket is MISSING after teardown"
  warn "restoring it"
  run sudo -n systemctl restart containerd.service 2>/dev/null || true
  sleep 2
  if sudo -n test -S "$SYS_CD_SOCK" 2>/dev/null; then
    ok "system containerd socket restored"
  else
    fail "still missing -- run: sudo systemctl restart containerd.service"
  fi
fi

if systemctl is-active --quiet containerd.service 2>/dev/null; then
  ok "containerd.service active"
else
  fail "containerd.service is NOT active -- investigate before leaving"
fi


if docker -H unix:///var/run/docker.sock info >/dev/null 2>&1; then
  n="$(docker -H unix:///var/run/docker.sock images -q 2>/dev/null | wc -l)"
  c="$(docker -H unix:///var/run/docker.sock ps -aq 2>/dev/null | wc -l)"
  ok "system daemon responding: ${n} images, ${c} containers -- untouched"
else
  fail "system daemon is not responding -- this is unexpected, investigate before leaving"
  exit 1
fi

printf '\n  Done. Unset the override in your shell if you set it:\n'
printf '      unset DOCKER_HOST\n\n'
