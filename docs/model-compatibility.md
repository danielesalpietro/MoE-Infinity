# Model Compatibility

This page lists the Mixture-of-Experts (MoE) model families MoE-Infinity
supports, how each is loaded, and known limitations around quantization and
multi-GPU execution. For the internal registration mechanism (module map,
execution paths), see [ARCHITECTURE.md](../ARCHITECTURE.md).

## How model support works

MoE-Infinity dispatches a checkpoint to an expert-offloading wrapper based on
the `architectures` field in the model's `config.json`. The mapping lives in
[`moe_infinity/common/constants.py`](../moe_infinity/common/constants.py):
`MODEL_MAPPING_NAMES` matches a substring of the architecture name (e.g.
`"mixtral"`, `"qwen3"`, `"deepseek"`) to a HuggingFace model class, and
`MODEL_MAPPING_TYPES` selects the expert-block layout used by the offloading
runtime. If a checkpoint's architecture isn't in `MODEL_MAPPING_NAMES`,
loading raises a `RuntimeError` listing the currently supported
architectures — that error message is always the authoritative list, this
page explains what it means in practice.

## Supported model families

The **Experts** column is `total experts (active per token)` for the example
checkpoint in that row — it's a checkpoint config value
(`num_local_experts`/`num_experts`/`n_routed_experts` + `num_experts_per_tok`,
depending on family), not a property of the architecture in general: other
checkpoints in the same family can use different expert counts (e.g.
GPT-OSS-20B vs. GPT-OSS-120B, below).

| Model family | Example checkpoints | Experts (active/total) | Loader | Notes |
|---|---|---|---|---|
| [DeepSeek-V2 / V3](https://huggingface.co/collections/deepseek-ai/deepseek-v2-669a1c8b8f2dbc203fbd7746) | `deepseek-ai/DeepSeek-V2-Lite-Chat`, `deepseek-ai/DeepSeek-V3` | V2-Lite: 64 (6) + 2 shared; V3: 256 (8) + 1 shared | HuggingFace (`DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`) | Reference full-precision path used in CI/e2e regression tests (`DeepSeek-V2-Lite-Chat`). |
| DeepSeek-V4-Flash | requires a `transformers` build shipping `DeepseekV4ForCausalLM` | 256 (6) routed, FP4 | HuggingFace class registered only when importable; full offload path uses the **DeepSeek-native** (non-HuggingFace) checkpoint format | FP4 (E2M1) routed experts + FP8 shared experts/attention. The HF-format model is skipped automatically if `DeepseekV4ForCausalLM` isn't available in your `transformers`. Full expert-offload support for the native checkpoint format is documented separately in [`moe_infinity/models/deepseek_v4/README.md`](../moe_infinity/models/deepseek_v4/README.md), including the native CUDA FP4 kernel path (Blackwell/SM120) and the `convert.py` checkpoint-prep step. |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Mixtral-8x22B` | 8 (2) | HuggingFace (`MixtralForCausalLM`) | The only model family with a **fused GPTQ expert-forward path**; see [Quantization support](#quantization-support). |
| [Qwen3-MoE](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `Qwen/Qwen3-30B-A3B` | 128 (8) | HuggingFace (`Qwen3MoeForCausalLM`) | Has a dedicated paged-attention implementation (`models/qwen3_paged_attention.py`) for the continuous-batching server. |
| [GPT-OSS](https://huggingface.co/models?search=gpt-oss) | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | 20b: 32 (4); 120b: 128 (4) | HuggingFace (`GptOssForCausalLM`) | Native MXFP4 expert quantization, with a Triton dequant/GEMM fallback when the checkpoint isn't natively MXFP4. |
| [DBRX](https://huggingface.co/models?search=dbrx) | `databricks/dbrx-instruct` | 16 (4) | HuggingFace (`DbrxForCausalLM`) | |
| [Jamba](https://huggingface.co/models?search=jamba) | `ai21labs/Jamba-v0.1` | 16 (2) | HuggingFace (`JambaForCausalLM`) | Hybrid Mamba/attention + MoE architecture; the MoE block only replaces the MLP in every other decoder layer. |
| [OLMoE](https://huggingface.co/models?search=olmoe) | `allenai/OLMoE-1B-7B-0924` | 64 (8) | HuggingFace (`OlmoeForCausalLM`) | |
| [Meta NLLB-MoE](https://huggingface.co/facebook/nllb-moe-54b) | `facebook/nllb-moe-54b` | 128 (2) | HuggingFace (`NllbMoeForConditionalGeneration`) | Encoder-decoder (translation); a sparse MoE layer replaces the FFN in 1 out of every 4 transformer blocks (`encoder_sparse_step=4`). |

`OPTForCausalLM` also appears in the internal mapping but is a dense (non-MoE)
model kept for legacy/testing purposes — it is not part of the supported MoE
model surface.

## Full MoE architecture coverage

MoE-Infinity's supported-model set has changed over time: architectures get
added when a contributor writes an offloading wrapper for them, and have
occasionally been **dropped** — either because they depended on vendored
(non-upstream) modeling code that became a maintenance burden, or because a
specific integration bug made them not worth maintaining. This table tracks
every MoE architecture that has been supported, is supported, or is a
plausible near-term candidate (i.e. it already exists as a HuggingFace
`transformers` model class), regardless of its current state.

This repository doesn't tag releases with semantic version numbers (the
package version has stayed `0.0.1` in `moe_infinity/__init__.py` across its
history), so "last supported in" below points at the last commit/PR where
the architecture was present, rather than a version number.

> **Upstream check (2026-09-12):** this fork tracks a point-in-time snapshot
> of [`EfficientMoE/MoE-Infinity`](https://github.com/EfficientMoE/MoE-Infinity).
> The upstream `main` branch has since moved to commit `6da092f` and added
> **Qwen3.5-MoE** and three **GLM** families (see the 🆙 rows below) that
> aren't in this fork's `moe_infinity/common/constants.py` yet. Upstream also
> replaced its own `docs/model-compatibility.md` with a different kind of
> document — a DFlash (speculative-decoding drafter) capability/evidence
> matrix (`validated` / `implemented-experimental` / `not recorded` per
> feature) rather than a simple supported/unsupported list — because that
> project now tracks per-feature validation state, not just "does it load".
> That format doesn't map onto this fork's actual code (no DFlash module
> exists here), so it isn't reproduced below; this table stays in the
> simpler supported/EoSupport/coming-soon shape that matches what's
> actually implemented in this repository.
>
> Upstream also fixed a latent bug relevant to anyone adding a new
> architecture here: `parse_expert_type` used to check `MODEL_MAPPING_NAMES`
> keys in insertion order, so a short key (`"qwen3"`) could shadow a longer,
> more specific one whose architecture string contains it as a substring
> (`"qwen3_5moeforconditionalgeneration"` also contains `"qwen3"`). Upstream
> now sorts candidate keys longest-first before matching. This fork's
> `constants.py` doesn't hit the bug today (none of its keys nest inside one
> another), but the fix is worth carrying over the moment a family like
> Qwen3.5 is added here.

The **Experts** column is `total (active per token)` for a representative
checkpoint of that architecture; it varies by checkpoint within a family
(see the callouts in the table above for GPT-OSS/DeepSeek).

| Architecture | HF class | Status | Experts (active/total) | Notes |
|---|---|---|---|---|
| DeepSeek-V2 | `DeepseekV2ForCausalLM` | ✅ Supported | 64 (6) + 2 shared | |
| DeepSeek-V3 | `DeepseekV3ForCausalLM` | ✅ Supported | 256 (8) + 1 shared | |
| DeepSeek-V4-Flash | `DeepseekV4ForCausalLM` (HF) / native checkpoint | ✅ Supported | 256 (6) routed, FP4 | Registered only when `transformers` ships `DeepseekV4ForCausalLM`; full offload path targets the native (non-HF) checkpoint format. |
| Mixtral | `MixtralForCausalLM` | ✅ Supported | 8 (2) | Only family with a fused GPTQ expert-forward kernel. |
| Qwen3-MoE | `Qwen3MoeForCausalLM` | ✅ Supported | 128 (8) | |
| GPT-OSS | `GptOssForCausalLM` | ✅ Supported | 32–128 (4) | 32 for the 20B checkpoint, 128 for 120B. Native MXFP4 expert path. |
| DBRX | `DbrxForCausalLM` | ✅ Supported | 16 (4) | |
| Jamba | `JambaForCausalLM` | ✅ Supported | 16 (2) | Hybrid Mamba/attention + MoE; MoE only in every other layer. |
| OLMoE | `OlmoeForCausalLM` | ✅ Supported | 64 (8) | |
| Meta NLLB-MoE | `NllbMoeForConditionalGeneration` | ✅ Supported | 128 (2) | Encoder-decoder. |
| Qwen3.5-MoE | `Qwen3_5MoeForConditionalGeneration` | 🆙 Upstream only | n/a (checkpoint-dependent) | Shipped upstream (PR #128, `Qwen/Qwen3.5-35B-A3B`, text-only expert offload); not yet in this fork's `constants.py`. |
| GLM-5.2 | `GlmMoeDsaForCausalLM` | 🆙 Upstream only | 256 (8) | Shipped upstream (`zai-org/GLM-5.2-FP8`); FP8 routed-expert offload, built-in MTP, no DFlash drafter. Not in this fork. |
| GLM-5.3 | `GlmMoeDsaForCausalLM` | 🆙 Upstream only | 256 (8) | Same registry entry/base as GLM-5.2 upstream (`zai-org/GLM-5.3`) — gains are post-training only. Not in this fork. |
| GLM-5.3-Flash | `Glm5NextForConditionalGeneration` | 🆙 Upstream only | 288 (8) + 1 shared | New upstream `glm5_next` family; hybrid KDA linear attention + DSA + mHC hyper-connections + vision tower stay resident, only routed experts offload. Not in this fork. |
| Snowflake Arctic | `ArcticForCausalLM` (vendored, not in upstream `transformers`) | ⛔ EoSupport | 128 (2) | Last supported at commit `9e42c9d` (PR #75). Removed at commit `f178db3` (PR #97, subsuming PR #90): "Arctic and Grok models are removed entirely. They are not available in upstream HuggingFace transformers." Users needing Arctic must pin to a pre-`f178db3` release. |
| xAI Grok-1 | `Grok1ModelForCausalLM` (vendored, not in upstream `transformers`) | ⛔ EoSupport | 8 (2) | Same removal as Arctic above — same commit, same reason (not in upstream `transformers`; ~1,161 lines of vendored modeling code dropped). |
| Google Switch Transformers | `SwitchTransformersForConditionalGeneration` (upstream, but dropped anyway) | ⛔ EoSupport | 8–2048 (1), checkpoint-dependent | Last supported at commit `9e42c9d` (PR #75). Removed at commit `f178db3` (PR #97): a C++ `ExpertDispatcher::OutputFunc` out-of-bounds indexing bug, the native engine's incompatibility with its encoder-decoder architecture, and the model being considered outdated. Unlike Arctic/Grok, this one *is* in upstream `transformers` — it was dropped for integration/maintenance reasons, not availability. |
| Qwen2-MoE | `Qwen2MoeForCausalLM` | 🔜 Coming soon | 60 (4) + 4 shared | Available upstream in `transformers`; no wrapper in any MoE-Infinity branch yet. |
| Llama 4 Scout | `Llama4ForCausalLM` | 🔜 Coming soon | 16 (1) + 1 shared | Available upstream in `transformers`; no wrapper yet. |
| Llama 4 Maverick | `Llama4ForCausalLM` | 🔜 Coming soon | 128 (1) + 1 shared | Same HF class as Scout, different expert count; no wrapper yet. |
| IBM Granite MoE | `GraniteMoeForCausalLM` / `GraniteMoeSharedForCausalLM` | 🔜 Coming soon | 32–40 (8), checkpoint-dependent | Available upstream in `transformers`; no wrapper yet. |
| JetMoE | `JetMoeForCausalLM` | 🔜 Coming soon | 8 (2) | Available upstream in `transformers`; no wrapper yet. Also mixes attention heads via the same MoE mechanism. |
| Phi-3.5-MoE | `PhimoeForCausalLM` | 🔜 Coming soon | 16 (2) | Available upstream in `transformers`; no wrapper yet. |

"Coming soon" here means the architecture is a realistic candidate because it
already has an upstream `transformers` class to wrap — not an official
roadmap commitment. "Upstream only" means the real `EfficientMoE/MoE-Infinity`
project already supports it; it's listed here so this fork's gap versus
upstream is visible, not as a promise it'll be ported. As with any new model
family, adding one goes through an issue first per
[CONTRIBUTING.md](../CONTRIBUTING.md); see
[Adding support for a new model](#adding-support-for-a-new-model) below.

## Quantization support

Quantization is detected from a checkpoint's `config.json`
(`quantization_config`) or its `quantize_config.json` /
`quant_config.json`, via [`moe_infinity/utils/quantization.py`](../moe_infinity/utils/quantization.py).

| Method | Status |
|---|---|
| **GPTQ** | Supported. Requires `optimum[gptq]` installed. Fused expert-forward kernel is implemented for **Mixtral only** (`models/mixtral.py`); other architectures load a GPTQ checkpoint but without that fused path. |
| **AWQ** | Supported. Requires `autoawq` installed. Validated end-to-end on Mixtral (`tests/python/e2e/test_quantized_e2e.py`). |
| **MXFP4 / FP4** | Supported natively for GPT-OSS and DeepSeek-V4-Flash, with host-memory offloading of the quantized experts. Handled by a separate runtime path, not the generic GPTQ/AWQ detector. |
| **HQQ** | Not supported — loading raises a `ValueError` pointing you to the full-precision or GPTQ/AWQ variant. |
| **bitsandbytes** | Not supported — same fail-fast behavior as HQQ. |
| **GGUF** | Not supported — use llama.cpp/Ollama, or the full-precision/GPTQ/AWQ variant with MoE-Infinity. |
| **EXL2** | Not supported — use ExLlamaV2, or the full-precision/GPTQ/AWQ variant with MoE-Infinity. |

Unsupported formats fail fast with a descriptive error at load time rather
than silently producing incorrect output.

## Multi-GPU

Single-server multi-GPU inference is supported for all model families in the
table above: expert parameters are distributed round-robin across all
visible GPUs, with per-GPU expert caching, peer-to-peer transfers, and
dedicated I/O threads. Multi-node distributed inference (across separate
machines) and expert parallelism are not yet supported — see the
[Release Plan](../README.md#release-plan) in the main README for roadmap
status.

## Adding support for a new model

MoE-Infinity's model support is added by implementing a
`Sync<Model>MoeBlock`-style expert wrapper and registering it in
`common/constants.py` and `runtime/model_offload.py`. See
["Adding a new MoE model" in ARCHITECTURE.md](../ARCHITECTURE.md#7-where-to-look-when-) for the exact files to touch, and open an issue first per
[CONTRIBUTING.md](../CONTRIBUTING.md) to align on scope.
