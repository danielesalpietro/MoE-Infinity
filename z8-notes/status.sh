#!/usr/bin/env bash
# One short command that answers "where were we?" after the laptop slept.
#
# Long work on this host runs detached (setsid nohup) precisely so that a
# closed lid does not stop it. What dies is the SSH session watching it, so
# what is needed on return is a fast, complete snapshot -- not a tail of a log
# that may be minutes stale.
#
# Deliberately cheap: no du over the HF cache (slow, and it understates
# progress because Xet stages chunks before materialising them). Download
# progress is read from the container's own network counter instead.
echo "=== $(date -u '+%F %H:%M:%S') UTC · $(hostname) · up $(uptime -p | sed 's/^up //') ==="

echo "--- servizi ---"
printf "  %s\n" "$(systemctl is-active docker containerd vastai cron 2>/dev/null | paste -sd' ' | sed 's/^/docker containerd vastai cron: /')"

echo "--- stack ---"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q moe-infinity-server; then
  docker ps --filter name=moe-infinity --format "  {{.Names}}  {{.Status}}" 2>/dev/null
  echo "  modello: $(grep -E '^MOE_MODEL=' /mnt/pmem_emh2/MoE-Infinity/.env 2>/dev/null | cut -d= -f2-)"
  echo "  health:  $(curl -sS --max-time 4 http://192.168.1.110:8700/health 2>/dev/null || echo 'nessuna risposta')"
else
  echo "  non in esecuzione"
fi

echo "--- download in corso ---"
found=0
for c in $(docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'hf-pull'); do
  found=1
  net=$(docker stats --no-stream --format '{{.NetIO}}' "$c" 2>/dev/null)
  echo "  $c: ricevuti $net  (il traffico e piu attendibile del du: Xet fa staging)"
done
[ "$found" = 0 ] && echo "  nessuno"
for l in /tmp/hf-pull-*.log /tmp/moe-build*.log; do
  [ -f "$l" ] || continue
  grep -aqE "FINITO|Built|BUILD done" "$l" 2>/dev/null && echo "  $(basename "$l"): completato"
done

echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'

echo "--- spazio ---"
df -h /mnt/pmem_emh2 /mnt/wdc-docker 2>/dev/null | tail -2 | awk '{printf "  %-18s %s usati, %s liberi (%s)\n", $6, $3, $4, $5}'

echo "--- dove eravamo ---"
echo "  todolist:  ~/moe-infinity/todolist.md"
echo "  logbook:   ~/moe-infinity/logbook/$(ls -1 ~/moe-infinity/logbook/logbook_s*.md 2>/dev/null | sort | tail -1 | xargs -r basename)"
echo "  indagine:  ~/moe-infinity/logbook/logbook_issue123.md"
