# Benchmarks

Benchmark suites for evaluating model families and infrastructure components in vLLM-Omni, organized by modality. See per-directory READMEs for detailed usage.

## Directory layout

```
benchmarks/
├── tts/              Text-to-speech serving benchmarks + model-specific tests
├── diffusion/        Image/video generation serving benchmarks + model-specific tests
├── accuracy/         Image generation/editing quality benchmarks
├── distributed/      RDMA connector transfer tests
└── kernels/          Kernel-level micro-benchmarks and auto-tuners
```

## Benchmark families

### [TTS](tts/README.md) — Text-to-Speech

Model-agnostic serving benchmarks for TTS models, including Qwen3-TTS, VoxCPM2, Higgs-Audio, and MOSS-TTS variants.

- **Serving benchmark**: `tts/bench_tts.py` (CLI wrapper around `vllm bench serve --omni`)
- **Model registry**: `tts/model_configs.yaml` (add new models here, no code changes)
- **Datasets**: `tts/datasets/` (bundled smoke/design prompt sets, download instructions for full Seed-TTS corpus)
- **Model-specific**: `tts/fish-speech/` (Fish Speech DAC-code cache benchmark, async benchmark utils)
- **Key metrics**: TTFP, E2E latency, RTF, audio throughput, WER/SIM/UTMOS (optional quality)

### [Diffusion](diffusion/README.md) — Image and Video Generation

Online-serving benchmarks for diffusion image/video models, sending requests to the configured vLLM serving endpoint (`/v1/chat/completions`, `/v1/images/generations`, `/v1/images/edits`, or `/v1/videos`).

- **Serving benchmark**: `diffusion/diffusion_benchmark_serving.py` (async, multi-endpoint)
- **Backends**: `diffusion/backends.py` (shared request/response dataclasses and async HTTP clients)
- **Model-specific**: `diffusion/glm-image/` (GLM-Image T2I/I2I: HuggingFace baseline, offline, and online serving benchmarks)
- **Diagnostics**: `diffusion/bench_attention_backends.py` (attention kernel comparison), `diffusion/quantization_quality.py` (LPIPS quality loss from quantization)
- **Performance dashboards**: `diffusion/performance_dashboard/` (reference results for Qwen-Image, Wan2.2)
- **Key metrics**: request throughput, latency percentiles, SLO attainment, per-stage durations

### [Accuracy](accuracy/README.md) — Image Generation and Editing Quality

Accuracy benchmarks for image generation/editing models, adapting external suites to vLLM-Omni serving and local judge-evaluation flows.

- **Layout**: `accuracy/text_to_image/` (GEBench), `accuracy/image_to_image/` (GEdit-Bench)
- **Method**: generation and judge scoring both run through local `vllm-omni serve` endpoints

### [Distributed](distributed/omni_connectors/README.md) — RDMA Connector Testing

RDMA environment setup and transfer tests for `MooncakeTransferEngineConnector`, including pytest-based single-node checks and manual cross-node benchmarks.

- **Transfer modes**: `copy`, `zerocopy`, `gpu` (GPUDirect)
- **Supports**: single-node pytest suites and manual multi-node/cross-node transfer testing

### [Kernels](kernels/README.md) — Kernel Micro-Benchmarks

Kernel-level micro-benchmarks and auto-tuners for custom operators.

- **MoT GEMM**: `kernels/mot_linear_benchmarks.py` (Triton kernel auto-tuner for Mixture-of-Tokens GEMM operations)

### Common serving metrics framework

`vllm_omni/benchmarks/` extends `vllm bench serve --omni` with Omni-specific datasets, backends, and multimodal metrics. Key metrics include:

- **Text output**: TTFT (time to first token), TPOT (time per output token), ITL (inter-token latency)
- **Audio output**: TTFP (time to first audio packet), E2E latency, RTF (real-time factor)
- **Throughput**: request throughput, output token throughput, total token throughput, audio throughput

See `vllm_omni/benchmarks/serve.py` for the `vllm bench serve --omni` runner wrapper and `vllm_omni/benchmarks/metrics/` for Omni metric definitions.

## Adding a new benchmark

1. Identify the modality: `tts/`, `diffusion/`, or create a new top-level modality directory.
2. For model-specific benchmarks within an existing modality, create a subdirectory under the modality (e.g., `tts/fish-speech/`, `diffusion/glm-image/`).
3. Include a `README.md` with: purpose, prerequisites, usage examples, CLI arguments table, and key metrics.
4. If comparing against another runtime, use clear backend subfolders (e.g., `huggingface/`, `vllm-omni/`).
5. Place datasets and prompt files under the modality's `datasets/` directory if applicable.
6. Update this README with a link to the new benchmark.
