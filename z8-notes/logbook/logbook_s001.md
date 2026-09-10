# Logbook s001 — 2026-09-10 (UTC)

Host: `berlin-3eie` · repo `danielesalpietro/MoE-Infinity`, branch `feat/z8-serve-webui`

Continues [logbook_s000](logbook_s000.md).

## Objective

## Done

## Incidents

## Open at end of day

## Objective

Migrate the Docker data-root off `sda`, which has developed unrecoverable read
errors, and onto the freshly formatted `sdc`. Secondary benefit: kaalia's image
budget is 10% of the disk, so it goes from 46.6 GB to ~93 GB.

## Hardware fault — sda is failing

```
ata1.00: error: { UNC }
sd 0:0:0:0: [sda] Sense Key : Medium Error [current]
Add. Sense: Unrecovered read error - auto reallocate failed
I/O error, dev sda, sector 497427392 op 0x0:(READ)
```

`WDC WD5000AAKS-4`, serial `WD-WCAWFD232275`, 465.8 GB, SATA, from around
2008. **"auto reallocate failed"** means the drive could not remap the bad
sector — it is out of spares or the area is unwritable. This is not a
transient glitch.

### The one corrupted file

Exactly one file out of 1,859,628 is unreadable. `dd` returns **0 bytes** for
it after ~2 s of retries.

```
/mnt/wdc-docker/docker/containerd/io.containerd.snapshotter.v1.overlayfs/
  snapshots/4834/fs/usr/local/lib/python3.12/dist-packages/
  torch/profiler/__pycache__/_memory_profiler.cpython-312.pyc
```

| | |
|---|---|
| Size | 69,750 bytes |
| Mtime | 2026-08-30 05:28 |
| Readable | **no** — 0 bytes recovered |
| Physical sector | `497427392` on `/dev/sda` |
| Snapshot | `4834`, 2.0 GB |
| Image fingerprint | `torch 2.11.0+cpu`, `python3.12`, `/usr/local/lib/python3.12/dist-packages` (Debian-style, **not** the conda layout the MoE-Infinity image uses) |

Everything else under `torch/profiler/` in that snapshot reads fine, so the
damage is confined to this one file's sectors.

**Impact is low and self-healing.** A `.pyc` is a Python bytecode cache:
CPython regenerates it from the `.py` on next import. It is not source, not
data, and nothing is lost permanently. The only visible symptom would be an
`ImportError` or a stale-cache warning inside a container built on snapshot
4834 — and only until something rewrites it.

**For whoever owns that image:** the affected layer is a CPU-only PyTorch
2.11.0 environment on Python 3.12. Rebuilding that image, or simply deleting
the `__pycache__` directory inside it, resolves it. There is no need to treat
it as data loss.

## Migration status

| | |
|---|---|
| `sdc` formatted | XFS, `ftype=1`, `crc=1` (both required for overlay2) |
| Full copy | 260 GB, 1,859,628 files |
| Delta resync | 88 files, 2.17 GB, on a quiesced host |
| Alignment | source 1,859,628 / destination 1,859,627 — **difference of exactly 1**, the unreadable `.pyc` |
| `fstab` | untouched, so a reboot still brings back `sda` |
| Backup | 14 GB on NVMe: logbooks, scripts, repo, `.env`, evidence, HF cache |

Host quiesced for the verification: `vastai.service` stopped (0 kaalia
processes), docker and containerd stopped, single mount line in `/proc/mounts`.

## Incident — a false rollback, and a stacked mount

The first swap attempt rolled back on a check of mine that was wrong, then
left the host in a messy state. Both worth recording.

**The false alarm.** The check compared container counts for equality:
32 before, 33 after, reported as "containers lost". They were not lost — one
appeared. kaalia runs `docker run --rm vastai/test:...` every few minutes
(986 invocations in its log), so the count is never stable. On a host with an
active agent, the right check is that no container present *before* has
disappeared, not that the total matches.

**The stacked mount.** The rollback unmounted while Docker was already
running again, so the unmount failed on a busy filesystem — and `|| true`
swallowed the error. `mount /dev/sda1` then stacked on top, leaving three
layers:

```
/mnt/wdc-docker            xfs      /dev/sdc1     <- bottom
├─ .../rootfs/overlayfs/…  overlay  overlay       <- leftover container
└─ /mnt/wdc-docker         xfs      /dev/sda1     <- top, the one in use
```

The overlay layer was left by a container kaalia started *while the root was
sdc1* — which also explains the 33rd container. Unwound top-down and remounted
cleanly. Lessons applied: stop Docker before unmounting, use `umount -R`, and
never swallow an unmount failure.

Side note, and mild evidence for the copy: the snapshotter successfully
assembled layers from the copied store to start that container.

## Migration completed

```
root: /mnt/wdc-docker/docker -> /dev/sdc1
immagini: 50   container: 32   mancanti: 0 / 0
hello-world: OK
postgres:16:  OK -- postgres (PostgreSQL) 16.15 (Debian 16.15-1.pgdg13+2)
scrittura overlay: OK
/dev/sdc1  xfs  932G  266G used  667G free  29%
```

`postgres:16` starting is the answer to the question `getfattr` could not
settle: those layers were copied with `rsync -aHAX` and the snapshotter
assembled and ran them. The overlay xattrs survived — established
functionally rather than by inspection.

**kaalia recomputed its budget against the new disk, as predicted:**

```
update_image_cache() diskspace:931.1GB maximg_space:93.1GB totimg_space:42.9GB
potature: 0
```

46.6 GB -> **93.1 GB**. Headroom goes from 3.7 GB to ~50 GB, which is what
makes a single-daemon setup viable: the MoE-Infinity image (10.9 GB content
size) now fits without being pruned, so the isolated second daemon is no
longer needed. Nothing was pruned after the restore.

Unexpected recovery: the copy predates kaalia's deletion of
`real-esrgan/ssh`, so **that 18.3 GB image is back** — sdc has 50 images
where sda had 49.

## Incident — two rollbacks caused by a test that never ran

Both earlier swap attempts rolled back on:

```
timeout: failed to run command 'd': No such file or directory
```

`d()` was a shell function wrapping `docker -H ...`, and `timeout` cannot
invoke a shell function — it looks for a binary. So
`timeout 120 d run --rm hello-world` always failed, the grep found nothing,
and the check reported FAILED. **hello-world never failed; the swap worked on
the first attempt.**

The cost was mostly time, with one real consequence: the 08:23 rollback ran
while kaalia was alive, and that is the window in which it deleted
`real-esrgan/ssh`.

Lesson: a check that cannot distinguish "the thing failed" from "the check
failed" is worse than no check, because it triggers destructive recovery on
its own malfunction. Every docker invocation in the final script is the full
command.

## Quiescing this host

`systemctl stop vastai` holds for about two minutes. The watchdog is not
systemd's — it is cron:

```
/etc/cron.d/vastai_restart_everything
* * * * *  vastai_kaalia  bash /var/lib/vastai_kaalia/latest/restart.sh
```

```bash
if test `find "$HOME/kaalia.log" -mmin +1`; then
    sudo systemctl stop vastai; sleep 5; sudo systemctl start vastai
fi
```

A liveness check on the log file's mtime, every minute — 512 runs that day.
And because the unit declares `Wants=docker.service`, restarting vastai also
pulls Docker up.

`systemctl mask` does **not** work here: the unit is a real file in
`/etc/systemd/system/`, and systemd refuses to mask over it. The way to
quiesce is **stop cron first, then stop vastai** — with cron down the watchdog
cannot fire.

Restore afterwards: `systemctl start cron.service` then
`systemctl start vastai.service`.

A suspect cleared: the daily `configure_nft.py` job fails every day with
`No such file or directory` — the file does not exist — so it is not what
removed `docker0`.

## Decisions

- **`sda` will be physically replaced** with a 1-2 TB disk as soon as
  practical. It has a sector it cannot remap; it gets no more data. Until
  then it stays unmounted, holding the pre-migration copy as a cold rollback.

## Next

1. **`fstab`** — still untouched, so **a reboot remounts `sda`** and Docker
   silently goes back to the failing disk. Replace the `/mnt/wdc-docker` line
   with sdc1's UUID, keeping the old one commented. Then reboot as the real
   test.
2. Only after that, retire `sda`.
3. When the replacement disk arrives it can take over as the backup target,
   freeing `nvme0n1` for the GPUDirect work.
4. Clean up `/mnt/wdc-docker/docker-moe` (38 GB): the isolated daemon is not
   needed now that the budget is 93 GB.
5. Back to the actual investigation: the DeepSeek-V2-Lite guardrail download
   never ran — the script targeted the system daemon without `DOCKER_HOST`
   and failed on the broken `docker0`.

## Single daemon, and what it required

The isolated daemon is gone. `moe-infinity-serve:sm86` was moved into the
system daemon's store by starting **only `containerd-moe`** — never
`dockerd-moe` — exporting with `ctr` from the `moby` namespace, stopping it,
and `docker load`ing the 11 GB tar. `docker0` was checked before, during and
after: unchanged. That confirms it is `dockerd`, not `containerd`, that
rewrites the host-global nftables table.

```
export 2m41s (11 GB tar) | import 5m19s | system images 50 -> 51
kaalia prunings during the import: 0
test: import ok, torch 2.9.1+cu128
```

`03-uninstall-daemon.sh --purge` then reclaimed 37 GB (703 GB free on sdc).

**Nothing had to be reorganised on the data side.** The HF cache, the offload
directory and the Open WebUI database are bind mounts on PMEM, not Docker
volumes, so they are daemon-agnostic and came through untouched. Had they been
Docker volumes they would have been inside the isolated store and would each
have needed exporting. Worth keeping as a habit.

Corrected afterwards: `.env.z8.example` still set
`MOE_DOCKER_SOCK=/run/docker-moe.sock`. Compose would have bind-mounted a path
Docker then creates as a *directory*, and docker-proxy would have failed in a
way that reads as a dashboard bug. Now `/var/run/docker.sock` (commit
`75ef08a`). The daemon scripts are archived under
`~/moe-infinity/archive/isolated-daemon-2026-09/` with a note saying not to
run them and why.

## Open: restore /mnt/backup

`nvme0n1p1` is mounted at `/mnt/backup` for this work. It is **not in fstab**,
so a reboot unmounts it and returns the disk to GPUDirect Storage on its own.
The original GDS artefacts (`cufile_p2pdma.json`, `testfile.bin`) were never
touched.

Its contents are all redundant *now*: `home-moe-infinity` duplicates
`~/moe-infinity`, `hf-cache` duplicates the PMEM copy, and the 11 GB image tar
duplicates the image now in the system daemon. It can be cleared once the
investigation is confirmed working — not before, since the tar is the only
thing standing between us and a one-hour rebuild.

**Before clearing it:** the logbooks are not in git and existed only on this
host and in that backup. A third copy now sits on the Windows workstation at
`<repo>/.z8-logbook/`, git-ignored — the same reasoning as `CLAUDE.md`: they
describe this host and the fork is public.
