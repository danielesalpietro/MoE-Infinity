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
