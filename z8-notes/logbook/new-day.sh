#!/usr/bin/env bash
# Create the next logbook entry. The day counter is mechanical so it cannot
# drift: it is derived from the highest existing file, never typed by hand.
#
#   ./new-day.sh          create the next entry for today (UTC)
#   ./new-day.sh --show   print the current entry's path and date
set -euo pipefail
cd "$(dirname "$0")"

last="$(ls -1 logbook_s[0-9][0-9][0-9].md 2>/dev/null | sort | tail -1)"
if [ -z "$last" ]; then
  next="000"
else
  n="${last#logbook_s}"; n="${n%.md}"
  next="$(printf "%03d" $((10#$n + 1)))"
fi

if [ "${1:-}" = "--show" ]; then
  echo "current: ${last:-none}"
  echo "next:    logbook_s${next}.md"
  exit 0
fi

# Days are counted in UTC -- the Z8 runs on UTC, and every timestamp we cite
# (build logs, journalctl, the kaalia log) is in that frame.
today="$(date -u +%Y-%m-%d)"
last_date="$(head -1 "${last:-/dev/null}" 2>/dev/null | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2}" || true)"
if [ "$last_date" = "$today" ]; then
  echo "warning: ${last} is already dated ${today} (UTC)."
  echo "         A new entry is for a new day -- append to that file instead,"
  echo "         or pass --force if you really want a second entry today."
  [ "${2:-}" = "--force" ] || exit 1
fi

f="logbook_s${next}.md"
cat > "$f" <<TPL
# Logbook s${next} — ${today} (UTC)

Host: \`$(hostname)\` · repo \`danielesalpietro/MoE-Infinity\`, branch \`$(cd /mnt/pmem_emh2/MoE-Infinity 2>/dev/null && git branch --show-current 2>/dev/null || echo "?")\`

Continues [logbook_s$(printf "%03d" $((10#$next - 1)))](logbook_s$(printf "%03d" $((10#$next - 1))).md).

## Objective

## Done

## Incidents

## Open at end of day

TPL
echo "created $f"
