# Archived 2026-09-10

These scripts built and tore down a **second, isolated Docker daemon**
(`docker-moe` + `containerd-moe`) so our images would live outside the store
kaalia prunes. They worked, and they are archived rather than deleted because
the reasoning in them is still the record of why this host is awkward.

**Do not run them as they stand.** Two dockerds on one host are not safe under
Docker 29: its nftables rules live in a host-global `docker-bridges` table, so
the second daemon's startup cleanup removed `docker0` and left the system
daemon unable to start any container. That happened here on 2026-09-09.

The problem they solved is gone. kaalia's image budget is 10% of the Docker
disk; moving the data-root from the 466 GB `sda` to the 931 GB `sdc` raised it
from 46.6 GB to 93.1 GB, leaving ~50 GB of headroom. One daemon is enough.

Still true and worth keeping from them:

- `systemctl stop vastai` holds for about two minutes.
  `/etc/cron.d/vastai_restart_everything` runs every minute and restarts it
  whenever `kaalia.log` goes stale. **Stop cron first**, then vastai.
- `systemctl mask` does not work on `vastai.service`: it is a real file in
  `/etc/systemd/system/` and systemd refuses to mask over it.
- A mount transition makes kaalia read `diskspace:-0.0GB`, drop its budget to
  16 GB and start pruning. Quiesce before touching mounts.
- `containerd` alone does **not** touch nftables — only `dockerd` does. That
  is why the image could be exported safely with only `containerd-moe`
  running.

See `../logbook/logbook_s000.md` and `logbook_s001.md` for the full account.
