# adflux-inference

<div align="center">

![Adflux](https://img.shields.io/badge/Adflux-Inference%20Engine-FF4D00?style=for-the-badge)
![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT%209.x-76B900?style=for-the-badge&logo=nvidia)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)

**TensorRT-optimized inference pipeline for real-time AI video generation.**

[Core Engine](https://github.com/getadflux/adflux-core) · [Docs](https://github.com/getadflux/adflux-docs) · [Website](https://adflux.dev)

</div>

---

## Overview

`adflux-inference` is the production inference serving layer for the Adflux platform. It wraps the core video diffusion model with a TensorRT-optimized serving pipeline, exposing a high-performance REST API for real-time video generation requests from agency clients.

Designed for low-latency, high-throughput serving on NVIDIA GPU infrastructure — achieving sub-30 second generation times for 1080p video ads.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  ADFLUX INFERENCE SERVER                  │
├──────────────────────────────────────────────────────────┤
│                                                           │
│   Agency Client Request (REST API)                        │
│         │                                                 │
│         ▼                                                 │
│   ┌───────────────┐                                      │
│   │  FastAPI Gate  │  ◀── Auth / Rate Limiting           │
│   └───────┬───────┘                                      │
│           │                                               │
│           ▼                                               │
│   ┌───────────────┐                                      │
│   │ Request Queue  │  ◀── Redis Queue / Priority         │
│   └───────┬───────┘                                      │
│           │                                               │
│           ▼                                               │
│   ┌────────────────────────────────────┐                 │
│   │        TensorRT Inference Engine   │                 │
│   │                                    │                 │
│   │   ┌─────────┐    ┌─────────────┐  │                 │
│   │   │ Text    │    │   Video     │  │                 │
│   │   │ Encoder │───▶│  Diffusion  │  │                 │
│   │   │ (FP16)  │    │  UNet (FP8) │  │                 │
│   │   └─────────┘    └──────┬──────┘  │                 │
│   │                         │         │                 │
│   │                  ┌──────▼──────┐  │                 │
│   │                  │  VAE Decoder│  │                 │
│   │                  │   (FP16)    │  │                 │
│   │                  └──────┬──────┘  │                 │
│   └─────────────────────────┼─────────┘                 │
│                              │                           │
│                              ▼                           │
│                    ┌──────────────────┐                  │
│                    │  Video Encoder   │                  │
│                    │  H.264 / H.265   │                  │
│                    └────────┬─────────┘                  │
│                             │                            │
│                             ▼                            │
│                    ┌──────────────────┐                  │
│                    │   S3 / CDN Upload │                  │
│                    │   Webhook notify  │                  │
│                    └──────────────────┘                  │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## API Reference

### Generate Video

```http
POST /v1/generate
Authorization: Bearer {api_key}
Content-Type: application/json
```

```json
{
  "prompt": "A luxury skincare product on a marble surface, soft natural light, slow cinematic zoom",
  "duration": 10,
  "resolution": "1080p",
  "aspect_ratios": ["16:9", "9:16", "1:1"],
  "num_variants": 3,
  "brand_kit_id": "bk_abc123",
  "webhook_url": "https://agency.com/webhooks/adflux"
}
```

**Response:**

```json
{
  "job_id": "job_7f3a9c2d",
  "status": "queued",
  "estimated_time_seconds": 45,
  "variants_requested": 3
}
```

### Check Job Status

```http
GET /v1/jobs/{job_id}
```

```json
{
  "job_id": "job_7f3a9c2d",
  "status": "completed",
  "generation_time_seconds": 38.2,
  "variants": [
    {
      "id": "var_001",
      "format": "16:9",
      "resolution": "1080p",
      "url": "https://cdn.adflux.dev/outputs/job_7f3a9c2d/var_001.mp4",
      "duration": 10.0
    }
  ]
}
```

---

## TensorRT Optimization

We use NVIDIA TensorRT to optimize our inference pipeline for maximum throughput and minimum latency:

```python
from adflux_inference.tensorrt import AdfluxTRTEngine

# Export PyTorch model to TensorRT
engine = AdfluxTRTEngine.from_pytorch(
    model_path="checkpoints/adflux_v1",
    precision="fp8",           # FP8 for UNet, FP16 for VAE
    optimization_level=5,      # Maximum TensorRT optimization
    workspace_size_gb=16,
    target_gpu="H100"
)

engine.save("engines/adflux_v1_h100_fp8.trt")

# Benchmark
results = engine.benchmark(
    batch_sizes=[1, 2, 4],
    input_resolution=(1080, 1920),
    num_frames=25,
    num_runs=100
)
print(results.summary())
```

**Latency improvements with TensorRT:**

| Model | Baseline (PyTorch) | TensorRT FP16 | TensorRT FP8 |
|---|---|---|---|
| Text Encoder | 180ms | 42ms | 28ms |
| UNet (per step) | 1,200ms | 380ms | 210ms |
| VAE Decoder | 850ms | 220ms | 220ms |
| **Full Pipeline** | **~90s** | **~38s** | **~22s** |

---

## Deployment

### Docker (NVIDIA Container)

```bash
# Pull base image
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# Build Adflux inference container
docker build -t adflux-inference:latest .

# Run on NVIDIA GPU
docker run --gpus all \
  -p 8000:8000 \
  -e MODEL_PATH=/models/adflux_v1 \
  -e TENSORRT_ENGINE=/engines/adflux_v1_h100_fp8.trt \
  -v /models:/models \
  -v /engines:/engines \
  adflux-inference:latest
```

### Requirements

```
NVIDIA GPU (A10G / A100 / H100)
CUDA 12.x
TensorRT 9.x
Docker with NVIDIA Container Toolkit
16GB+ GPU VRAM
```

---

## Performance

| GPU | Concurrent Requests | P50 Latency | P99 Latency | Throughput |
|---|---|---|---|---|
| NVIDIA H100 SXM | 4 | 22s | 31s | 10.9 videos/min |
| NVIDIA A100 80GB | 2 | 38s | 52s | 3.1 videos/min |
| NVIDIA A10G 24GB | 1 | 58s | 78s | 1.0 videos/min |

---

## About Adflux

Adflux is an AI video generation platform purpose-built for marketing agencies. NVIDIA Inception member company.

**Founders:** Rohan Kapoor (CEO) · Dev Patel (AI Developer)

🌐 [adflux.dev](https://adflux.dev) · ✉️ dev@adflux.dev
