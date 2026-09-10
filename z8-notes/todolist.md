# Todo — deferred work

Things we decided to do and deliberately did **not** do yet. The logbooks
record what happened; this records what is owed.

---

## The gate

> **Close one before opening another.**

No new GitHub issue or PR while one of ours is still open. The point is not
bookkeeping: every session of this project has turned up something worth
filing, and filing them all produces a wall of open items that nobody reads
and that makes the next real finding invisible.

Two honest caveats on the rule, so it does not get gamed:

- Closing something **only** to earn the right to open the next thing is
  cheating. A close has to stand on its own merits.
- A finding that is genuinely urgent — data loss, a security problem, a wrong
  result being published — jumps the queue. Nothing currently in it qualifies.

## Gate state

Ours, still open upstream (`EfficientMoE/MoE-Infinity`):

| | | Why still open |
|---|---|---|
| [#123](https://github.com/EfficientMoE/MoE-Infinity/issues/123) | issue, OLMoE inference crash | **Reproduced 2026-09-09.** Not closed: the fix is not merged, and #205 alone does not resolve it |
| [#119](https://github.com/EfficientMoE/MoE-Infinity/pull/119) | PR, OLMoE support + silent init failures | Superseded by #205. **Should be closed on its own merits** — that is independent of this gate |

On the fork (`danielesalpietro/MoE-Infinity`): [#2](https://github.com/danielesalpietro/MoE-Infinity/issues/2) RunPod package, [#3](https://github.com/danielesalpietro/MoE-Infinity/issues/3) Blackwell profile. Both roadmap, neither blocking.

**So: nothing new gets opened until #119 is closed and #123 has an outcome.**

---

## Queue

### 1. Second OLMoE bug — transformers v5 contract  · deferred by the gate

With `MODEL_MAPPING_TYPES["olmoe"] = 5` the fused-kernel crash is gone and the
engine dies one frame later:

```
transformers/models/olmoe/modeling_olmoe.py:402  hidden_states = residual + hidden_states
TypeError: unsupported operand type(s) for +: 'Tensor' and 'tuple'
```

`moe_infinity/models/olmoe.py` returns `(hidden_states, router_logits)` — the
v4 contract — while v5's decoder layer consumes `self.mlp(...)` as a bare
tensor. Independent of #123, deserves its own issue.

*Unblocks when:* #119 closed, #123 resolved. *Before filing:* the scope across
the other nine wrappers is unverified — static comparison says mixtral, jamba,
dbrx, gpt_oss and nllb_moe return tuples too, but only OLMoE has been run.
Already mentioned publicly in the #206 comment, so it is not a secret; it just
does not have its own tracker yet.

### 2. OLMoE 4 → 5 PR  · promised publicly, waiting on the positive control

Flip the constant plus a one-line update to
`test_parse_expert_type_new_models`. Promised on #205 on 2026-09-09.

*Unblocks when:* a model demonstrably serves on this stack, so a failure can be
attributed. *Before filing:* check whether #205 merged — it changes the base.

### 3. ~~Positive control~~  · DONE 2026-09-10

`deepseek-ai/DeepSeek-V2-Lite-Chat` serves. Prefill and the first decode steps
are correct (" Paris", " Rome", " Celsius"). The prediction held and no
`Tensor + tuple` appeared. A failure on OLMoE is now attributable to OLMoE.

### 3b. ~~Third bug~~ — already filed as #191  · NOT OURS

Generation degenerating after ~3 tokens on DeepSeek-V2-Lite is
[#191](https://github.com/EfficientMoE/MoE-Infinity/issues/191), open since
2026-09-02, with fix [#195](https://github.com/EfficientMoE/MoE-Infinity/pull/195)
open and `DIRTY`. DeepSeek-V2/V3 specific (MLA paged-attention shim missing
outside tests). Nothing to file.

If anything is owed here it is a **confirmation comment on #191** — an
independent reproduction on different hardware — and that costs nothing
against the gate because it opens nothing. Low value while #195 is already
written, so: only if asked.

### 4. Close #119  · no blocker, just not done

Superseded by #205, 286 commits behind, still open in our name. This is the
close that most obviously stands on its own.

---

## Not issues — Z8 operations

By convention these stay in the logbook, never GitHub (this host is not the
project's business):

- **`BUILD_JOBS` build arg.** `-j$(nproc)` spawns 152 compilers on 32 cores,
  saturates the box and makes vast.ai's GPU probe fail, delisting the machine.
  Every rebuild will do it again.
- **Replace `sda`.** Unrecoverable read error at sector 497427392,
  `auto reallocate failed`. Unmounted, out of fstab, awaiting a 1–2 TB disk.
- **Clear `/mnt/backup`** and return `nvme0n1` to the GPUDirect tests. Not in
  fstab, so a reboot does it. Not before the investigation is confirmed — the
  11 GB image tar is the only thing between a mishap and a one-hour rebuild.
- **Reboot once** to prove the new `fstab`. `mount -a` was verified; a boot
  was not.
- **DHCP reservation or dynamic DNS** for the Z8. The public IP moved three
  times in a few hours, taking every SSH watcher with it.

## Upstream observations with no home yet

Noted while reading the code; not filed, and below the bar to jump the gate:

- `HOST_MEMORY_RATIO` is a compile-time `#define` in
  `core/memory/memory_pool.h`, not a runtime flag. On a 235 GB host that is a
  188 GB pool you cannot tune without rebuilding.
- `MOEINF_PIN_SIZE` and `MOEINF_SHM_SIZE` have no defaults and no
  documentation, and are `DLOG_FATAL` if the corresponding allocator starts
  without them.
- `api_server_v2` requires `max_tokens` on every request; Open WebUI does not
  send it by default, so the chat UI returns a 400 out of the box.
- `failed to bind layered paged KV store: paged backend store is not
  initialized` appears at startup. Unexplained, apparently harmless.
