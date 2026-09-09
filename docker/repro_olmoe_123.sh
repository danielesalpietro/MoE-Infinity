#!/usr/bin/env bash
# Reproduce EfficientMoE/MoE-Infinity#123 against a real GPU, then verify the
# proposed fix -- on the same host, same image, same checkpoint, changing
# exactly one integer between the two runs.
#
# Why this exists: the discussion on #205 established the layout bug by
# reading source and comparing shapes, but nobody on the thread has actually
# exercised the fused CUDA path. The failing TORCH_CHECK is in device code,
# so the only way to settle it is to run it.
#
# The claim under test
# --------------------
# MODEL_MAPPING_TYPES["olmoe"] = 4 routes OLMoE through the Mixtral branch of
# MoEMLP::ForwardHelper (core/parallel/expert_module.cpp), which reads the
# per-expert weight blob as [gate, down, up]. OLMoE's HF module registers
# [gate_proj, up_proj, down_proj], so up and down arrive swapped and
# fused_moe_ffn_into gets a down_proj of shape [I, H] where it wants [H, I].
# On allenai/OLMoE-1B-7B-0924-Instruct (H=2048, I=1024) that fails the
# H == H_out check at extensions/kernel/fused_moe_mlp.cu.
#
#   PASS for phase "before" = the crash reproduces, with that exact message.
#   PASS for phase "after"  = same request returns a completion.
#
# A phase "before" that does NOT crash is the interesting outcome: it would
# mean the analysis on #123 is wrong, or incomplete, and the PR should not be
# opened as written. Do not paper over it.
#
# Usage:
#   docker/repro_olmoe_123.sh before     # expect the crash
#   docker/repro_olmoe_123.sh patch      # flip olmoe 4 -> 5 in the container
#   docker/repro_olmoe_123.sh after      # expect a completion
#   docker/repro_olmoe_123.sh report     # print the collected evidence
#
# Run it from the repo root on the host running the stack.

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.webui.yml}"
SERVICE="moe-infinity"
CONTAINER="moe-infinity-server"
EVIDENCE_DIR="${EVIDENCE_DIR:-./.repro-123}"
MODEL="${MOE_MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}"
# Long enough for a cold load of a 7B checkpoint off an already-warm HF cache.
READY_TIMEOUT="${READY_TIMEOUT:-900}"

CONSTANTS_PATH="/workspace/MoE-Infinity/moe_infinity/common/constants.py"

mkdir -p "$EVIDENCE_DIR"

api_addr() {
  # Ask docker where the API actually landed, address included.
  #
  # Taking only the port and assuming 127.0.0.1 is wrong whenever the stack
  # publishes on a specific interface: this stack binds the host's LAN address,
  # so every probe failed with "Failed to connect to 127.0.0.1" and told us
  # nothing about the server. A probe that cannot reach the server is not
  # evidence either way.
  local mapping
  mapping="$(docker port "$CONTAINER" 8000/tcp 2>/dev/null | head -1)"
  [ -n "$mapping" ] || return 1
  case "$mapping" in
    0.0.0.0:*|"[::]:"*) echo "127.0.0.1:${mapping##*:}" ;;
    *) echo "$mapping" ;;
  esac
}

expert_type_in_container() {
  docker exec "$CONTAINER" python -c \
    'from moe_infinity.common.constants import MODEL_MAPPING_TYPES as m; print(m["olmoe"])'
}

wait_ready() {
  local addr deadline body
  addr="$(api_addr)" || { echo "cannot determine the published address" >&2; return 1; }
  deadline=$(( $(date +%s) + READY_TIMEOUT ))
  echo "waiting for the server on ${addr} (up to ${READY_TIMEOUT}s)..."
  while [ "$(date +%s)" -lt "$deadline" ]; do
    # No -f: a 503 still carries a body, and the body is the whole point.
    body="$(curl -sS --max-time 5 "http://${addr}/health" 2>/dev/null || true)"

    case "$body" in
      *'"status":"healthy"'*)
        echo "server healthy"
        return 0 ;;
      *'engine loop failed'*)
        # A failed engine loop never recovers -- /health keeps answering, just
        # with a reason. Treating that as "still loading" burns the whole
        # timeout on a question that is already settled, which is exactly what
        # the first run did: 15 minutes waiting for a server that was dead.
        echo "the engine loop has failed -- not waiting further:"
        printf '  %s\n' "$body"
        return 1 ;;
    esac

    # A container that has died is not going to become healthy either; fail
    # fast so a crash during *load* is not misread as a slow load.
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
      echo "container is no longer running" >&2
      return 1
    fi
    sleep 5
  done
  echo "timed out waiting for /health" >&2
  return 1
}

probe() {
  # One short, deterministic completion. Greedy and capped: we are asking
  # whether the expert forward runs at all, not measuring quality.
  local addr
  addr="$(api_addr)"
  curl -fsS --max-time 120 "http://${addr}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
          \"model\": \"${MODEL}\",
          \"messages\": [{\"role\": \"user\", \"content\": \"Name three primary colors.\"}],
          \"max_tokens\": 32,
          \"temperature\": 0
        }"
}

phase_before() {
  echo "== phase: before (expecting the crash) =="
  echo "olmoe expert type in container: $(expert_type_in_container)"
  wait_ready || true
  probe > "$EVIDENCE_DIR/before.response.json" 2> "$EVIDENCE_DIR/before.curl.err" \
    && echo "request returned a response (see before.response.json)" \
    || echo "request failed (see before.curl.err) -- expected here"
  # The TORCH_CHECK message surfaces in the server log, not the HTTP body.
  docker logs "$CONTAINER" > "$EVIDENCE_DIR/before.server.log" 2>&1
  echo
  echo "-- expert-forward failures in the log --"
  grep -cE "fused_moe_ffn_into|expert forward failed" "$EVIDENCE_DIR/before.server.log" \
    || echo "0 (this is the interesting outcome -- read the note at the top)"
  grep -m3 -E "hidden dim mismatch|expert forward failed" "$EVIDENCE_DIR/before.server.log" || true
}

phase_patch() {
  echo "== phase: patch (olmoe 4 -> 5) =="
  docker exec "$CONTAINER" python - <<'PY'
import re, pathlib
p = pathlib.Path("/workspace/MoE-Infinity/moe_infinity/common/constants.py")
src = p.read_text()
new, n = re.subn(r'("olmoe"\s*:\s*)4', r"\g<1>5", src)
if n != 1:
    raise SystemExit(f'expected exactly one "olmoe": 4 to rewrite, found {n}')
p.write_text(new)
print("rewrote", p)
PY
  # The mapping is read at model-registration time, so the process has to be
  # restarted -- an edit alone changes nothing for an already-loaded model.
  docker compose -f "$COMPOSE_FILE" restart "$SERVICE"
  echo "olmoe expert type after restart: $(expert_type_in_container)"
}

phase_after() {
  echo "== phase: after (expecting a completion) =="
  echo "olmoe expert type in container: $(expert_type_in_container)"
  wait_ready
  probe > "$EVIDENCE_DIR/after.response.json" 2> "$EVIDENCE_DIR/after.curl.err" \
    && echo "request returned a response" \
    || { echo "request still failing -- the flip is not sufficient" >&2; }
  docker logs "$CONTAINER" > "$EVIDENCE_DIR/after.server.log" 2>&1
  echo
  echo "-- expert-forward failures in the log --"
  grep -cE "fused_moe_ffn_into|expert forward failed" "$EVIDENCE_DIR/after.server.log" || true
  echo "-- completion text --"
  python3 -c "
import json,sys
try:
    d = json.load(open('$EVIDENCE_DIR/after.response.json'))
    print(d['choices'][0]['message']['content'])
except Exception as e:
    print('no parseable completion:', e)
" 2>/dev/null || cat "$EVIDENCE_DIR/after.response.json"
}

phase_report() {
  echo "=================== evidence ==================="
  echo "host:      $(hostname)"
  echo "gpu:       $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | paste -sd'; ')"
  echo "model:     $MODEL"
  echo "image:     $(docker inspect -f '{{.Config.Image}}' "$CONTAINER" 2>/dev/null)"
  echo "commit:    $(git rev-parse --short HEAD 2>/dev/null)"
  echo
  for phase in before after; do
    log="$EVIDENCE_DIR/${phase}.server.log"
    [ -f "$log" ] || continue
    echo "--- $phase ---"
    echo "expert forward failures: $(grep -cE 'fused_moe_ffn_into|expert forward failed' "$log" || true)"
    grep -m2 -E "hidden dim mismatch" "$log" || echo "(no hidden-dim mismatch logged)"
    echo
  done
}

case "${1:-}" in
  before) phase_before ;;
  patch)  phase_patch ;;
  after)  phase_after ;;
  report) phase_report ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
