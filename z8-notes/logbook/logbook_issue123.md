# Issue #123 — OLMoE crashes on inference (`fused_moe_ffn_into: hidden dim mismatch`)

Tracking file for one problem, across days. The daily logbooks record what
happened on a given date; this records the state of *this* investigation.

## References

**Upstream — `EfficientMoE/MoE-Infinity`**

| | Opened (UTC) | By | State |
|---|---|---|---|
| [#119](https://github.com/EfficientMoE/MoE-Infinity/pull/119) *Fix OLMoE support and silent model-init failures* | 2026-07-18 18:10:39 | danielesalpietro | open, superseded — should be closed |
| [#123](https://github.com/EfficientMoE/MoE-Infinity/issues/123) *[BUG] OLMoE crashes on inference* | **2026-07-19 23:19:34** | danielesalpietro | open, no maintainer response |
| [#205](https://github.com/EfficientMoE/MoE-Infinity/pull/205) *register OLMoE in MoE param / expert-id parsing* | 2026-09-07 08:39:30 | drunkcoding | open, CI green, no review, `BLOCKED` |

#123 was reproduced on real hardware on **2026-09-09**, 52 days after it was
filed, and still with no maintainer response on the issue.

- #205's single commit: `fd1a5e8eefb6` (2026-09-07 08:39:07 UTC), co-authored
  by danielesalpietro
- Branch base: `upstream/main` at **`02a4afebffa4`** — *feat: opt-in INT8
  KV-cache quantization (#183)*, 2026-09-06

**Fork — `danielesalpietro/MoE-Infinity`**, branch `feat/z8-serve-webui`

| | |
|---|---|
| Code baked into the image under test | **`c5100128e836`** — *test(repro): scripted reproduction of the OLMoE inference crash (#123)*, 2026-09-09 22:28:16 +0200 |
| Reproduction script | `docker/repro_olmoe_123.sh` |

Two commits landed *after* the image was built and are **not** in it —
`b8993bd` (compose: docker-proxy socket) and `3828cab` (gitignore CLAUDE.md).
Both touch only compose/env files, which are read at `up` time rather than
baked by `COPY . .`, so the running code is `c5100128e836` exactly.

**Image and runtime under test**

| | |
|---|---|
| Image | `moe-infinity-serve:sm86` |
| Digest | `sha256:35bbdfc3c3a007f078a83e23256d81e5a877d3fd49f8b3c96c913238f098d814` |
| Built | 2026-09-09 22:31:17 UTC |
| torch / CUDA | 2.9.1+cu128 / 12.8 |
| transformers | 5.17.0 |
| CUTLASS | 3.9.2, `CUTLASS_NVCC_ARCHS=86` |

---

## The claim

`MODEL_MAPPING_TYPES["olmoe"] = 4` routes OLMoE through the **Mixtral** branch
of `MoEMLP::ForwardHelper` (`core/parallel/expert_module.cpp:281-285`), which
reads the per-expert weight blob as `[gate, down, up]`:

```cpp
auto& gate_proj = param_[0];
auto& up_proj   = (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE) ? param_[1] : param_[2];
auto& down_proj = (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE) ? param_[2] : param_[1];
```

OLMoE's HF module registers `[gate_proj, up_proj, down_proj]` — the Qwen3
layout, which is type **5**. So `up` and `down` arrive swapped, and
`fused_moe_ffn_into` receives a `down_proj` of shape `[I, H]` where it wants
`[H, I]`, failing `TORCH_CHECK(H == H_out)` at
`extensions/kernel/fused_moe_mlp.cu:123`.

`param_` is populated **by index** (`MoEMLP::SetTensorsFromIds`), from
`named_parameters()` order — not by name. Note that `MoEMLP` never consults
`ExpertTraits`: the two are parallel representations, and only the index
convention in `ForwardHelper` governs the fused path.

## Status

| | |
|---|---|
| Crash reproduced on real hardware | **yes — 2026-09-09** |
| Fix (4 → 5) verified to resolve it | **yes — 2026-09-09** |
| OLMoE serves end to end after the fix | **no** — a second, distinct bug is underneath |
| Positive control | **obtained 2026-09-10** — see below |
| Jamba affected the same way | unverified, `mfethe1` has a CPU probe |
| Numerical correctness after the fix | out of scope — this is a layout contract, not a numerics test |

---

## 2026-09-09 — reproduction confirmed

Environment: Z8 G4 `berlin-3eie`, RTX 3090 (sm_86, 24 GB), driver 595.84,
image `moe-infinity-serve:sm86` on an isolated Docker daemon (see References
above for digests and versions). Model `allenai/OLMoE-1B-7B-0924-Instruct` —
`hidden_size` 2048, `intermediate_size` 1024, 64 experts, 16 layers, bfloat16.

The code under test is `c5100128e836`, the commit the image was built from —
**not** the branch tip at run time. The two later commits changed only compose
and env files.

`MODEL_MAPPING_TYPES["olmoe"]` left at the buggy `4`, verified inside the
image before the run. The only library change carried was the two-line
`olmoe` parsing branch in `hf_config.py` — the substance of #205, which is
still unmerged and without which the checkpoint does not load at all.

**Sequence.** The model loads. `/health` returns `{"status":"healthy"}` at
23:05:53 UTC. The engine loop then starts, the first expert forward runs, and
the process goes permanently unhealthy:

```
{"status":"unhealthy","reason":"engine loop failed: fused_moe_ffn_into: hidden dim mismatch"}
```

This is exactly the shape #123 describes: loading succeeds, inference does not.

**Traceback** — the chain lands precisely on the path under analysis:

```
transformers/models/olmoe/modeling_olmoe.py:401   hidden_states = self.mlp(hidden_states)
moe_infinity/models/olmoe.py:64                   self.expert_executor.wait_dispatch_local()
moe_infinity/distributed/expert_executor.py:460   self.expert_dispatcher.wait_expert()
RuntimeError: fused_moe_ffn_into: hidden dim mismatch
```

Two details that strengthen the result:

- **No request is needed.** The engine fails on its own at startup, so this is
  not an input-dependent edge case.
- **The signature is unambiguous.** Earlier the same day, a degenerate
  smoke-test checkpoint (`vprovorg/tiny-random-Mixtral-8x7B-v0.1`, random
  weights, 6x16 matrices) failed generation with a plain
  `RuntimeError: size mismatch` inside torch's `F.linear`. That is a different
  failure in Python. This one is the `TORCH_CHECK` in device code, with the
  kernel's own message. There is no attribution ambiguity between the two.

Evidence: `.repro-123/before.server.log` (10 KB) in the repo checkout on the Z8.

### Caveat, stated plainly

There is **no positive control**. The intended smoke test was meant to prove
that generation works in general before blaming OLMoE, but the tiny random
checkpoint has degenerate dimensions and cannot generate at all. So the run
proves that OLMoE with `olmoe = 4` fails with this specific kernel check — it
does not independently prove that the stack generates correctly for a working
model. The `after` phase supplies that control: same image, same host, one
integer changed.

---

## 2026-09-09 — fix verified, and a second bug underneath

Same host, same image, same checkpoint. One integer changed inside the running
container (`docker exec` rewrite of `constants.py`) followed by a service
restart, verified by reading the constant back: `olmoe expert type after
restart: 5`.

**The fused-kernel crash is gone.** Counted over the log window after the
patched restart (23:46:14 UTC):

```
occurrences of fused_moe_ffn_into after the flip: 0
```

against two before it, at 23:15:22 and 23:30:20, both with `olmoe = 4`. The
analysis on #123 is confirmed end to end, on hardware.

**But OLMoE still does not serve.** The engine now dies further along, on an
error that the kernel check had been masking:

```
transformers/models/olmoe/modeling_olmoe.py:401   hidden_states = self.mlp(hidden_states)
transformers/models/olmoe/modeling_olmoe.py:402   hidden_states = residual + hidden_states
TypeError: unsupported operand type(s) for +: 'Tensor' and 'tuple'
```

`moe_infinity/models/olmoe.py` returns `final_hidden_states, router_logits` —
the transformers v4 contract. In v5, `OlmoeDecoderLayer.forward` consumes
`self.mlp(...)` as a bare Tensor. The wrapper returns a tuple; the layer adds
it to a residual; it fails.

The 4→5 fix does not cause this. It **uncovers** it: with `olmoe = 4` the
`TORCH_CHECK` aborted upstream of this line, so the second defect was
unreachable. No amount of source reading would have found it, because the
first bug hid it.

Scope not established: all ten wrappers under `moe_infinity/models/` reference
`router_logits`, so the same v4-contract return may be general. Only OLMoE was
verified — whether the others break depends on how each architecture's decoder
layer changed in v5. A hypothesis to check, not a result.

Environment for this run is unchanged from the reproduction above; script
commit `62781c1`, transformers 5.17.0.

### A defect in our own reporting

`phase_report` counted matches across the whole of `docker logs`, which
accumulates across restarts. It reported "expert forward failures: 2" for the
after phase and the run was very nearly written up as *the flip is not
sufficient*. Only the timestamps showed both failures predated the patched
restart. The counts must be scoped to the window after the last
`Started server process`.

## Next

1. **Fix the health gate in `repro_olmoe_123.sh`.** It waits 900 s for
   `/health`, which never returns once the engine loop dies. It should
   recognise `{"status":"unhealthy","reason":"engine loop failed: ..."}` and
   proceed immediately. Cost so far: 15 minutes of waiting on a result that
   was already decided.
2. ~~`patch` phase~~ — done, verified.
3. ~~`after` phase~~ — done. The kernel crash is gone; a second bug blocks
   serving, so there is still **no positive control** showing a completion.
4. **Fix `phase_report`** to count only the window after the last restart.
5. **Decide how to report the second bug.** It is a transformers v5
   compatibility defect in the OLMoE wrapper, independent of #123, and
   deserves its own issue — with the caveat that its scope across the other
   nine wrappers is unverified.
5. **Open the PR** promised on #205: `MODEL_MAPPING_TYPES["olmoe"] = 5` plus
   the one-line update to `test_parse_expert_type_new_models`
   (`tests/python/unit/test_model_registry.py`, currently asserts
   `OlmoeForCausalLM -> 4`). Check first whether #205 has merged, since it
   changes the base.
6. **Comment on #123 and #205** with the runtime evidence — the thing neither
   party to that discussion has.
7. **Close #119**, superseded by #205.

## Notes for the write-up

- The assertion being changed, `("OlmoeForCausalLM", 4)`, was introduced by
  `193b7eb` — the same commit that registered DBRX, OLMoE and Jamba and chose
  the values. It pins the initial guess, not a verified behaviour, which is
  why it stayed green on a wrong answer.
- `mfethe1` proposed a layout-derived regression test instead of another
  constant assertion, and flagged Jamba. Agreed split: OLMoE 4→5 ours; Jamba
  theirs; table-wide layout test theirs, after the first two, with DBRX carved
  out (`SyncDbrxFFNBlock` wraps `DbrxExperts` with fused `w1`/`v1`/`w2`, not
  three per-expert `nn.Linear`s, so a probe cannot walk it the same way).
- A caution worth repeating upstream: a test that reads `ExpertTraits`
  weight names would be validating the wrong artifact, since `MoEMLP` does not
  consult them.
- Unrelated error seen in the same log, worth reporting separately if it
  persists: `failed to bind layered paged KV store: RuntimeError('paged
  backend store is not initialized')`.

## 2026-09-09 — looking for a positive control

The investigation still has no model that demonstrably serves, so no failure
can be attributed with confidence. Before downloading anything, a static check
of the return contract across all wrappers, against what each transformers v5
decoder layer consumes:

| Wrapper | Returns | v5 layer wants | Prediction |
|---|---|---|---|
| olmoe | tuple | tensor | broken *(verified)* |
| mixtral | tuple | tensor | **broken** |
| jamba | tuple | tensor | **broken** |
| dbrx, nllb_moe, gpt_oss | tuple | different pattern | undetermined |
| deepseek, deepseek_v2, deepseek_v3 | tensor | tensor | **works** |
| qwen (qwen3), qwen3_5 | tensor | tensor | **works** |

(`glm_moe_dsa` reads as a tuple only because of a comma inside
`view(bsz, seq, hid)` — it returns a tensor.)

This says the guardrail must come from the DeepSeek or Qwen3 families. Mixtral
would have been the instinctive choice and would have failed for the same
reason as OLMoE, which would have been read as "the stack is broken" rather
than "this wrapper is broken".

Chosen: **`deepseek-ai/DeepSeek-V2-Lite-Chat`**, 31.4 GB, arch
`DeepseekV2ForCausalLM`, registered as `deepseek` type 5 — also the default
`MOE_MODEL` in `Dockerfile.serve`. Rejected: `Qwen/Qwen3-30B-A3B` (61.1 GB,
too slow on this link) and `Qwen/Qwen1.5-MoE-A2.7B-Chat` (arch
`Qwen2MoeForCausalLM` is not in `MODEL_MAPPING_NAMES` at all).

The prediction is falsifiable: if DeepSeek-V2-Lite also fails with
`Tensor + tuple`, the contract analysis is wrong. If it serves, we have both a
positive control and independent evidence that the v5 tuple return is the
OLMoE-side defect.

Note for reproducibility: `constants.py` inside the **running container** now
has `olmoe = 5` (edited live). The image still ships 4, so a `down` + `up`
recreates the container and reverts it; a `restart` preserves it.

## 2026-09-10 — positive control obtained

`deepseek-ai/DeepSeek-V2-Lite-Chat` (31.4 GB, 64 experts, 6 per token, 2
shared, `TOPO: 56 stages, 26 sparse`) loads and **serves** on the same host,
same image, same settings. Only the model differs from the OLMoE runs.

```
max_tokens=1, temperature=0
  "The capital of France is"      -> " Paris"
  "The capital of Italy is"       -> " Rome"
  "Water freezes at zero degrees" -> " Celsius"
  "The color of the sky is"       -> " a"
```

Those come out of a full forward over the whole prompt, through the expert
path. **The stack computes correctly.** The prediction held: DeepSeek's wrapper
returns a bare tensor and it serves, where OLMoE's returns a tuple and it does
not — and it failed with no `Tensor + tuple` anywhere.

**What this establishes:** a failure on OLMoE is now attributable to OLMoE
rather than to a broken environment. That was the missing piece.

**What it does not establish:** correctness of long-form generation. It
degenerates sharply after about three tokens:

| `max_tokens` | output |
|---|---|
| 2 | `" Paris,"` |
| 3 | `" Paris, "` |
| 5 | `" Paris, 10"` |
| 8 | `" Paris, 10000"` |

Prefill is right, the first couple of decode steps are right, then it collapses
into a run of zeros. That is a **third bug**, independent of both OLMoE ones.

**CORRECTION — this is not our finding.** It is
[#191](https://github.com/EfficientMoE/MoE-Infinity/issues/191), filed by
`drunkcoding` on 2026-09-02, eight days before this run, and it reports the
same outputs almost token for token:

| Prompt | #191 | ours |
|---|---|---|
| `The capital of France is` | `" Paris, 1000000000000"` | `" Paris, 10000"` |
| `2 + 2 =` | `" 100000000000000"` | `" 10000000000"` |

Root cause per that issue: the continuous-batching runner only wires the paged
KV cache into attention when it finds a module whose class name is a
registered `*PagedAttention`
(`model_runner.py::_get_paged_attention_classes` matches
`{"DeepseekV2PagedAttention", "DeepseekV3PagedAttention"}`). That shim exists
only in tests, so `use_paged_context` is false and the forward runs without
paged context. The fix is
[#195](https://github.com/EfficientMoE/MoE-Infinity/pull/195) (2026-09-03,
638+/235-, adds `moe_infinity/models/deepseek_v2_paged_attention.py`) — **open
and `DIRTY`**, i.e. conflicting with main.

Two things follow, and the second matters more:

- **It is DeepSeek-V2/V3 specific**, tied to MLA attention. It is *not*
  evidence of a general decode problem in the stack, which is how I first
  wrote it up.
- **It therefore strengthens the positive control rather than weakening it.**
  The degeneration is explained and confined to a known gap in DeepSeek's
  attention wiring; the expert path and the prefill are sound. Attribution of
  the OLMoE failures stands.

**How this was missed:** the PR list read at the very start of the session
included `195 drunkcoding feat(serving): DeepSeek-V2/V3 paged-attention shim
(MLA) — fixes #191`. The information was in hand for hours and was not
connected until the result was challenged. Read the open issues before
calling something a discovery.

### Side finding: MOE_DEVICE_MEMORY_RATIO

The first attempt at this test ran at `0.80` and OOMed *during inference*, not
at load: the model came up healthy at 6.8 GB resident, then the first
completion grew the expert cache to 24106 MiB of 24576 and the engine died in
`device_caching_allocator.cpp:28`. The ratio caps the **expert cache**, not
total VRAM. 0.80 was reasoned about with OLMoE (13.8 GB) in mind and does not
transfer to a 31.4 GB model. Back to 0.5.

Method note: that run changed two variables at once — the ratio and a model
2.3x larger — so it was inconclusive rather than informative. The retry varied
only the model.

### Correction: the KV-store warning is not the cause

I had written that `failed to bind layered paged KV store` was "the visible
edge" of #191. It is not. The same warning appears when serving
`yujiepan/qwen3-moe-tiny-random`, which does **not** collapse — 8 generated
tokens come out varied rather than as a repeated token. So the warning is a
separate and apparently harmless condition, and #191's cause is what its
author states: the `DeepseekV2PagedAttention` shim existing only in tests.

Correlation was taken for causation on a single observation. The second
observation cost about a minute.

### qwen3 path probe (2026-09-10)

`yujiepan/qwen3-moe-tiny-random` — `Qwen3MoeForCausalLM`, bf16, hidden 64 /
moe_intermediate 128, 8 experts, 2 per token, 2 layers, GQA. Unlike the tiny
Mixtral (degenerate 6x16, fp16, rejected by the BF16-only fused kernel) its
dimensions are small but sane.

Loads in 25 s, serves at 1, 8 and 24 tokens, engine stays healthy, no
fused-kernel error. Random weights, so this proves **execution of the qwen3
path**, not correctness — the output cannot be judged. What it does show is
the absence of the #191 collapse signature.

That is enough to justify fetching `Qwen/Qwen3-30B-A3B` in bf16 (56.9 GB) as
the real anchor: same fused BF16 expert path as OLMoE, GQA rather than MLA so
outside #191, and a wrapper that returns a bare tensor.
