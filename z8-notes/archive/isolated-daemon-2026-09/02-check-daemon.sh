#!/usr/bin/env bash
# Verify the isolated daemon is up, correctly placed, GPU-capable, and
# genuinely invisible to vast.ai.
#
# Read-only: starts nothing, changes nothing. Exits non-zero if any check
# fails, so it can gate a build in a script.
#
#   ./02-check-daemon.sh          # full check, pulls a small image for the GPU test
#   ./02-check-daemon.sh --no-gpu # skip the GPU test (no pull, no container)

set -uo pipefail
cd "$(dirname "$0")"
# shellcheck source=_common.sh
source ./_common.sh

RUN_GPU_TEST=1
[ "${1:-}" = "--no-gpu" ] && RUN_GPU_TEST=0

failures=0
check_failed() { fail "$*"; failures=$((failures + 1)); }

step "containerd"

if [ -f "$CONTAINERD_UNIT" ]; then ok "unit present: ${CONTAINERD_UNIT}"
else check_failed "containerd unit missing -- run ./01-install-daemon.sh"; fi

if containerd_active; then ok "${CONTAINERD_SERVICE} active"
else check_failed "${CONTAINERD_SERVICE} is not active"; fi

if containerd_responds; then ok "socket present: ${CONTAINERD_SOCK}"
else check_failed "no socket at ${CONTAINERD_SOCK}"; fi

# Our containerd must be on our own socket AND must not have taken over the
# host one. A version-4 config silently ignores the legacy [grpc] table, and
# an instance that falls back to defaults binds the SYSTEM socket path.
sys_cd_sock="/run/containerd/containerd.sock"
if sudo -n test -S "$sys_cd_sock" 2>/dev/null; then
  ok "system containerd socket exists (${sys_cd_sock})"
  # /proc/PID/cmdline is NUL-separated, so a grep for "--address <path>" with a
  # space in it can never match, and the earlier attempt to normalise it with
  # tr inside a command substitution lost the NULs before tr ever saw them.
  # Read it, translate the separators, and match with a shell glob -- no pipes,
  # no quoting games. The file is world-readable, so no sudo either.
  our_pid="$(systemctl show -p MainPID --value "$CONTAINERD_SERVICE" 2>/dev/null)"
  if [ -n "$our_pid" ] && [ "$our_pid" != "0" ] && [ -r "/proc/${our_pid}/cmdline" ]; then
    # xargs -0 splits on NUL and echo rejoins with spaces. tr inside a command
    # substitution did not survive the round trip -- the NULs reached bash,
    # which stripped them and warned, leaving the arguments glued together.
    cd_cmd="$(xargs -0 echo < "/proc/${our_pid}/cmdline" 2>/dev/null)"
    case "$cd_cmd" in
      *"--address ${CONTAINERD_SOCK}"*)
        ok "our containerd is pinned to ${CONTAINERD_SOCK}" ;;
      *)
        check_failed "our containerd is not pinned to ${CONTAINERD_SOCK}"
        info "cmdline: ${cd_cmd}" ;;
    esac

    # The decisive evidence: which socket is this pid actually listening on?
    if command -v ss >/dev/null 2>&1; then
      if sudo -n ss -lxp 2>/dev/null | grep -q "pid=${our_pid}.*"          && sudo -n ss -lxp 2>/dev/null | grep "pid=${our_pid}," | grep -q "$CONTAINERD_SOCK"; then
        ok "confirmed listening on ${CONTAINERD_SOCK} (pid ${our_pid})"
      fi
      if sudo -n ss -lxp 2>/dev/null | grep "pid=${our_pid}," | grep -q " /run/containerd/containerd.sock"; then
        check_failed "our containerd is ALSO listening on the host socket -- it has hijacked it"
      fi
    fi
  else
    warn "cannot read /proc/${our_pid}/cmdline -- skipping the pinning check"
  fi
else
  check_failed "the SYSTEM containerd socket is missing -- run: sudo systemctl restart containerd.service"
fi

step "Service"

if daemon_unit_exists; then ok "unit present: ${UNIT_PATH}"
else check_failed "unit missing -- run ./01-install-daemon.sh"; fi

if daemon_active; then ok "systemd reports ${SERVICE_NAME} active"
else check_failed "${SERVICE_NAME} is not active -- sudo systemctl start ${SERVICE_NAME}"; fi

if daemon_responds; then ok "daemon answers on ${DOCKER_SOCK}"
else check_failed "no response on ${DOCKER_SOCK}"; fi

if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
  warn "enabled at boot -- expected NOT enabled, so it only runs when you ask"
else
  ok "not enabled at boot (intentional)"
fi

# Everything below needs a live daemon.
if ! daemon_responds; then
  printf '\n%s%d check(s) failed%s\n' "$C_ERR" "$failures" "$C_OFF"
  exit 1
fi

step "Placement"

actual_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)"
if [ "$actual_root" = "$DATA_ROOT" ]; then
  ok "data-root is ${actual_root}"
else
  check_failed "data-root is ${actual_root}, expected ${DATA_ROOT}"
fi

root_fs="$(df --output=target "$actual_root" 2>/dev/null | tail -1)"
if [ "$root_fs" = "$STORE_MOUNT" ]; then
  ok "image store sits on ${STORE_MOUNT}, the disk meant for container storage"
else
  check_failed "image store is on ${root_fs}, expected ${STORE_MOUNT}"
fi

# Beside the system daemon root, never inside it.
case "$actual_root" in
  "$SYSTEM_DATA_ROOT"/*|"$SYSTEM_DATA_ROOT")
    check_failed "our data-root is INSIDE the system daemon (${SYSTEM_DATA_ROOT})" ;;
  *)
    ok "separate from the system daemon root (${SYSTEM_DATA_ROOT})" ;;
esac

avail="$(store_free_gb)"
if [ -n "$avail" ] && [ "$avail" -ge "$MIN_FREE_GB" ]; then
  ok "${avail} GB free on ${STORE_MOUNT}"
else
  warn "only ${avail:-?} GB free on ${STORE_MOUNT} (want >= ${MIN_FREE_GB}) -- shared with vast.ai renters"
fi

pmem_avail="$(pmem_free_gb)"
if [ -d "$WORKLOAD_ROOT" ]; then
  ok "workload data on PMEM: ${WORKLOAD_ROOT} ($(du -sh "$WORKLOAD_ROOT" 2>/dev/null | cut -f1)), ${pmem_avail:-?} GB free"
else
  warn "${WORKLOAD_ROOT} not found -- HF cache and offload dir are expected there"
fi

step "Image store ownership"

# The failure this check exists for: a dockerd started without an explicit
# --containerd silently attaches to the system containerd, and then our images
# show up in the system daemon -- visible to vast.ai, and prunable by it.
cmdline="$(dockerd_containerd_addr)"
if printf "%s" "$cmdline" | grep -q -- "--containerd"; then
  info "dockerd was started with an explicit --containerd flag"
fi

configured="$(sudo -n grep -oE "\"containerd\"[^,]*" "$CONFIG_FILE" 2>/dev/null | head -1)"
if printf "%s" "$configured" | grep -q "$CONTAINERD_SOCK"; then
  ok "daemon.json points at our containerd (${CONTAINERD_SOCK})"
else
  check_failed "daemon.json does not point at ${CONTAINERD_SOCK} -- images would be shared"
fi

if [ -d "${CONTAINERD_ROOT}/root" ]; then
  ok "containerd content store is ours: ${CONTAINERD_ROOT}/root ($(sudo -n du -sh "${CONTAINERD_ROOT}/root" 2>/dev/null | cut -f1))"
else
  warn "${CONTAINERD_ROOT}/root not created yet -- expected once an image is pulled"
fi

step "Runtime"

runtime_path="$(docker info --format '{{range $k,$v := .Runtimes}}{{if eq $k "nvidia"}}{{$v.Path}}{{end}}{{end}}' 2>/dev/null)"
if [ "$runtime_path" = "$REAL_NVIDIA_RUNTIME" ]; then
  ok "nvidia runtime is the genuine one (${runtime_path})"
elif [ -z "$runtime_path" ]; then
  check_failed "no nvidia runtime registered -- GPU containers will not start"
else
  check_failed "nvidia runtime points at ${runtime_path} (expected ${REAL_NVIDIA_RUNTIME})"
  info "the system daemon deliberately uses vast.ai's kaalia_docker_shim; this one must not"
fi

step "Isolation from vast.ai"

our_ids="$(docker images -q 2>/dev/null | sort -u)"
our_count="$(printf "%s
" "$our_ids" | grep -c . || true)"
sys_ids="$(docker -H unix:///var/run/docker.sock images -q 2>/dev/null | sort -u)"
sys_count="$(printf "%s
" "$sys_ids" | grep -c . || true)"

info "this daemon: ${our_count} image(s) | system daemon: ${sys_count} image(s)"

# Comparing image IDs alone cannot tell a shared store from the same public
# image pulled independently: content-addressed IDs are identical either way.
# So the failure condition is the two sets being *the same set*, which is what
# a shared containerd actually looks like -- not a non-empty intersection.
if [ "$our_count" -gt 0 ] && [ "$our_ids" = "$sys_ids" ]; then
  check_failed "both daemons list an identical image set -- the store is shared"
elif [ "$our_count" -eq 0 ]; then
  info "no images here yet -- isolation is structural; build something to see it proven"
else
  overlap="$(comm -12 <(printf "%s
" "$our_ids") <(printf "%s
" "$sys_ids") | grep -c . || true)"
  ok "image sets differ -- the stores are separate"
  [ "$overlap" -gt 0 ] && info "${overlap} id(s) appear in both: expected when the same public image is pulled twice"
fi

# The assertion that actually matters: nothing we build locally may show up on
# the system daemon, where vast.ai can see and prune it.
leaked="$(docker -H unix:///var/run/docker.sock images --format "{{.Repository}}:{{.Tag}}" 2>/dev/null           | grep -E "^moe-infinity" || true)"
if [ -n "$leaked" ]; then
  check_failed "locally built image(s) visible to the system daemon:"
  printf "       %s
" "$leaked"
else
  ok "no moe-infinity image is visible to the system daemon"
fi

if systemctl is-active --quiet vastai.service 2>/dev/null; then
  info "vastai.service is active -- fine, it only governs the system daemon"
else
  info "vastai.service is not active"
fi

step "GPU"

if [ "$RUN_GPU_TEST" -eq 0 ]; then
  info "skipped (--no-gpu)"
else
  # Show the whole host inventory first: a GPU missing from the stack is
  # usually a configuration choice, not a fault, and the two are easy to
  # confuse when only one card is ever mentioned.
  info "GPUs on this host:"
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,uuid              --format=csv,noheader 2>/dev/null | sed "s/^/         /"

  # Test the GPU the stack is actually configured to use, not blindly the
  # first one -- otherwise a stack pinned to GPU 1 gets a green light from a
  # test that exercised GPU 0.
  env_file="${PMEM_MOUNT}/MoE-Infinity/.env"
  gpu_target=""
  if [ -r "$env_file" ]; then
    gpu_target="$(grep -E "^MOE_GPU_DEVICE_IDS=" "$env_file" 2>/dev/null | cut -d= -f2- | tr -d "\"" )"
  fi
  if [ -n "$gpu_target" ]; then
    info "stack is pinned to: ${gpu_target}"
  else
    gpu_target="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)"
    warn "no MOE_GPU_DEVICE_IDS in ${env_file} -- falling back to the first GPU"
  fi

  if [ -z "$gpu_target" ]; then
    check_failed "no GPU to test"
  else
    if out="$(docker run --rm --runtime nvidia                 -e NVIDIA_VISIBLE_DEVICES="$gpu_target"                 -e NVIDIA_DRIVER_CAPABILITIES=utility                 ubuntu:22.04 nvidia-smi -L 2>&1)"; then
      ok "the configured GPU is reachable from a container in this daemon"
      printf "       %s
" "$out"

      # Arch sanity: the fused MoE kernels are built for one CUTLASS arch, so a
      # card of a different compute capability fails at runtime, not slowly.
      want_arch="$(grep -E "^CUTLASS_NVCC_ARCHS=" "$env_file" 2>/dev/null | cut -d= -f2- | tr -d "\"" )"
      got_cc="$(printf "%s" "$out" | grep -oE "GPU-[0-9a-f-]+" | head -1                 | xargs -I{} nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id={} 2>/dev/null                 | tr -d ".")"
      if [ -n "$want_arch" ] && [ -n "$got_cc" ]; then
        if [ "$want_arch" = "$got_cc" ]; then
          ok "compute capability ${got_cc} matches CUTLASS_NVCC_ARCHS=${want_arch}"
        else
          check_failed "GPU is sm_${got_cc} but the image is built for sm_${want_arch} -- kernels would fail at runtime"
        fi
      fi
    else
      check_failed "GPU test container failed"
      printf "       %s
" "$out" | head -5
    fi
  fi

  # Anything present but excluded is worth naming, so it reads as a decision.
  total_gpus="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | grep -c .)"
  if [ "${total_gpus:-0}" -gt 1 ]; then
    for u in $(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null); do
      case "$gpu_target" in
        *"$u"*|all) ;;
        *) info "not used by the stack: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader --id="$u" 2>/dev/null) (${u})" ;;
      esac
    done
  fi
fi

step "Result"

if [ "$failures" -eq 0 ]; then
  printf '%s  all checks passed  %s\n\n' "$C_OK" "$C_OFF"

  # Verified 2026-09-09: docker compose DOES honour --context (it is a CLI
  # plugin and inherits it). The export just saves repeating the flag.
  if docker context inspect moe >/dev/null 2>&1; then
    peek="docker --context moe ps"
  else
    peek="docker -H unix://${DOCKER_SOCK} ps"
  fi

  cat <<TXT

  Working with this daemon
  ------------------------

  1. Point the shell at it, so you can drop --context from every command:

       export DOCKER_HOST=unix://${DOCKER_SOCK}

  2. See what is running, without exporting anything:

       ${peek}

  3. Build and start the stack:

       cd ${PMEM_MOUNT}/MoE-Infinity
       export DOCKER_HOST=unix://${DOCKER_SOCK}
       docker compose -f docker-compose.webui.yml up -d --build

  With neither DOCKER_HOST nor --context moe you are talking to the SYSTEM
  daemon. Anything built there is visible to vast.ai and gets pruned -- that
  is how an hour of build was lost once already.

TXT
  echo
  exit 0
else
  printf '%s  %d check(s) failed  %s\n\n' "$C_ERR" "$failures" "$C_OFF"
  exit 1
fi
