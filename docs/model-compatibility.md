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

| Model family | Example checkpoints | Loader | Notes |
|---|---|---|---|
| [DeepSeek-V2 / V3](https://huggingface.co/collections/deepseek-ai/deepseek-v2-669a1c8b8f2dbc203fbd7746) | `deepseek-ai/DeepSeek-V2-Lite-Chat`, `deepseek-ai/DeepSeek-V3` | HuggingFace (`DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`) | Reference full-precision path used in CI/e2e regression tests (`DeepSeek-V2-Lite-Chat`). |
| DeepSeek-V4-Flash | requires a `transformers` build shipping `DeepseekV4ForCausalLM` | HuggingFace class registered only when importable; full offload path uses the **DeepSeek-native** (non-HuggingFace) checkpoint format | FP4 (E2M1) routed experts + FP8 shared experts/attention. The HF-format model is skipped automatically if `DeepseekV4ForCausalLM` isn't available in your `transformers`. Full expert-offload support for the native checkpoint format is documented separately in [`moe_infinity/models/deepseek_v4/README.md`](../moe_infinity/models/deepseek_v4/README.md), including the native CUDA FP4 kernel path (Blackwell/SM120) and the `convert.py` checkpoint-prep step. |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Mixtral-8x22B` | HuggingFace (`MixtralForCausalLM`) | The only model family with a **fused GPTQ expert-forward path**; see [Quantization support](#quantization-support). |
| [Qwen3-MoE](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `Qwen/Qwen3-30B-A3B` | HuggingFace (`Qwen3MoeForCausalLM`) | Has a dedicated paged-attention implementation (`models/qwen3_paged_attention.py`) for the continuous-batching server. |
| [GPT-OSS](https://huggingface.co/models?search=gpt-oss) | `openai/gpt-oss-*` | HuggingFace (`GptOssForCausalLM`) | Native MXFP4 expert quantization, with a Triton dequant/GEMM fallback when the checkpoint isn't natively MXFP4. |
| [DBRX](https://huggingface.co/models?search=dbrx) | `databricks/dbrx-instruct` | HuggingFace (`DbrxForCausalLM`) | |
| [Jamba](https://huggingface.co/models?search=jamba) | `ai21labs/Jamba-*` | HuggingFace (`JambaForCausalLM`) | Hybrid Mamba/attention + MoE architecture. |
| [OLMoE](https://huggingface.co/models?search=olmoe) | `allenai/OLMoE-*` | HuggingFace (`OlmoeForCausalLM`) | |
| [Meta NLLB-MoE](https://huggingface.co/facebook/nllb-moe-54b) | `facebook/nllb-moe-54b` | HuggingFace (`NllbMoeForConditionalGeneration`) | Encoder-decoder (translation), not a decoder-only causal LM like the rest of the table. |

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

| Architecture | HF class | Status | Notes |
|---|---|---|---|
| DeepSeek-V2 | `DeepseekV2ForCausalLM` | ✅ Supported | |
| DeepSeek-V3 | `DeepseekV3ForCausalLM` | ✅ Supported | |
| DeepSeek-V4-Flash | `DeepseekV4ForCausalLM` (HF) / native checkpoint | ✅ Supported | Registered only when `transformers` ships `DeepseekV4ForCausalLM`; full offload path targets the native (non-HF) checkpoint format. |
| Mixtral | `MixtralForCausalLM` | ✅ Supported | Only family with a fused GPTQ expert-forward kernel. |
| Qwen3-MoE | `Qwen3MoeForCausalLM` | ✅ Supported | |
| GPT-OSS | `GptOssForCausalLM` | ✅ Supported | Native MXFP4 expert path. |
| DBRX | `DbrxForCausalLM` | ✅ Supported | |
| Jamba | `JambaForCausalLM` | ✅ Supported | Hybrid Mamba/attention + MoE. |
| OLMoE | `OlmoeForCausalLM` | ✅ Supported | |
| Meta NLLB-MoE | `NllbMoeForConditionalGeneration` | ✅ Supported | Encoder-decoder. |
| Snowflake Arctic | `ArcticForCausalLM` (vendored, not in upstream `transformers`) | ⛔ EoSupport | Last supported at commit `9e42c9d` (PR #75). Removed at commit `f178db3` (PR #97, subsuming PR #90): "Arctic and Grok models are removed entirely. They are not available in upstream HuggingFace transformers." Users needing Arctic must pin to a pre-`f178db3` release. |
| xAI Grok-1 | `Grok1ModelForCausalLM` (vendored, not in upstream `transformers`) | ⛔ EoSupport | Same removal as Arctic above — same commit, same reason (not in upstream `transformers`; ~1,161 lines of vendored modeling code dropped). |
| Google Switch Transformers | `SwitchTransformersForConditionalGeneration` (upstream, but dropped anyway) | ⛔ EoSupport | Last supported at commit `9e42c9d` (PR #75). Removed at commit `f178db3` (PR #97): a C++ `ExpertDispatcher::OutputFunc` out-of-bounds indexing bug, the native engine's incompatibility with its encoder-decoder architecture, and the model being considered outdated. Unlike Arctic/Grok, this one *is* in upstream `transformers` — it was dropped for integration/maintenance reasons, not availability. |
| Qwen2-MoE | `Qwen2MoeForCausalLM` | 🔜 Coming soon | Available upstream in `transformers`; no wrapper yet in `moe_infinity/models/`. |
| Llama 4 (Scout / Maverick) | `Llama4ForCausalLM` | 🔜 Coming soon | Available upstream in `transformers`; no wrapper yet. |
| IBM Granite MoE | `GraniteMoeForCausalLM` / `GraniteMoeSharedForCausalLM` | 🔜 Coming soon | Available upstream in `transformers`; no wrapper yet. |
| JetMoE | `JetMoeForCausalLM` | 🔜 Coming soon | Available upstream in `transformers`; no wrapper yet. |
| Phi-3.5-MoE | `PhimoeForCausalLM` | 🔜 Coming soon | Available upstream in `transformers`; no wrapper yet. |

"Coming soon" here means the architecture is a realistic candidate because it
already has an upstream `transformers` class to wrap — not an official
roadmap commitment. As with any new model family, adding one goes through an
issue first per [CONTRIBUTING.md](../CONTRIBUTING.md); see
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
