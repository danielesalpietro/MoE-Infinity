# MoE-Infinity — isolated Docker daemon on the Z8

## Why

The system Docker daemon on this host is managed by vast.ai. Its `kaalia`
agent polls `docker images -q` every ~3 minutes and enforces an image-space
budget of ~46.6 GB (10% of `/mnt/wdc-docker`). On 2026-09-09 it deleted the
MoE-Infinity serving image **39 seconds after the build tagged it**:

    update_image_cache diskspace:465.5GB maximg_space:46.6GB totimg_space:53.1GB
      removing moe-infinity-serve:sm86: af6917b1...
    result: Untagged: moe-infinity-serve:sm86

Roughly an hour of build, discarded — and it would happen again on every
rebuild. These scripts stand up a second Docker daemon with its own socket
and its own image store, which kaalia does not enumerate and cannot prune.

Isolation comes from the **separate daemon and socket**, not from where the
layers sit: kaalia runs `docker images -q` against the system socket, and our
daemon is not on it. So the image store lives on `/mnt/wdc-docker`, the disk
meant for container storage, beside the system daemon root rather than
inside it. PMEM stays reserved for the workload that benefits from it -- the
HF cache, the expert offload directory and the chat database under
`/mnt/pmem_emh2/moe-infinity`.

## Scripts

| Script | Does |
|---|---|
| `01-install-daemon.sh` | Creates and starts the daemon. Idempotent — exits cleanly if already up. |
| `02-check-daemon.sh` | Read-only verification: service, placement, runtime, isolation, GPU. `--no-gpu` to skip the container test. |
| `03-uninstall-daemon.sh` | Removes it and restores the host. Keeps the image store unless `--purge`. |
| `_common.sh` | Shared settings. Sourced, not run. |

## Usage

    ~/moe-infinity/01-install-daemon.sh
    ~/moe-infinity/02-check-daemon.sh

    export DOCKER_HOST=unix:///run/docker-moe.sock
    cd /mnt/pmem_emh2/MoE-Infinity
    docker compose -f docker-compose.webui.yml build

**Every `docker` and `docker compose` command for this project must target
this daemon**, via `DOCKER_HOST` or `--context moe` (compose honours the
context -- verified). With neither, you are on the system daemon and anything
you build there gets reaped.

## Layout

| | System daemon (vast.ai) | This daemon |
|---|---|---|
| Socket | `/var/run/docker.sock` | `/run/docker-moe.sock` |
| Data-root | `/mnt/wdc-docker/docker` | `/mnt/wdc-docker/docker-moe/data` |
| `nvidia` runtime | `kaalia_docker_shim` (vast.ai) | `/usr/bin/nvidia-container-runtime` |
| Registry | `docker*.vast.ai` mirrors | direct |
| Bridge / subnets | `docker0`, 172.17–172.26 | `docker-moe0`, 172.30 / 172.31 |
| Visible to kaalia | yes | **no** |

Not enabled at boot — after a reboot, `sudo systemctl start docker-moe`.

## What these scripts never touch

The system daemon and its config, its containers, images and volumes (the
`northstream-*` and `caliper-*` stacks), vast.ai, and
`/mnt/pmem_emh2/moe-infinity` (HF cache, expert offload, chat database).

Do not run `docker system prune -a` or `docker volume prune` against the
system daemon on this host: they are host-wide and would take those stacks
out with them.


## Hazard: containerd config schema

containerd 2.3 moved the gRPC server into a plugin
(`io.containerd.server.v1.grpc`) and **ignores a top-level `[grpc]` table in a
version-4 config**. An instance configured that way does not fail -- it starts
happily on the containerd default address, which is the *system* socket
`/run/containerd/containerd.sock`. It unlinks and rebinds that path, and on
shutdown deletes it, leaving the host containerd running with no socket file.

That happened once here, on 2026-09-09. The host recovered with
`sudo systemctl restart containerd.service`.

The address, root and state are therefore passed as **command-line flags** in
`containerd-moe.service`, not set in the config file: flags take precedence
and are not subject to schema changes between versions. `01-install-daemon.sh`
additionally fingerprints the system socket's inode before and after starting
our instance, and aborts and restores if it moved.

If the system containerd socket ever goes missing:

    sudo systemctl restart containerd.service
