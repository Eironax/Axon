# Axon

Containerized LLM Latent Firewall & Activation Steering Engine.

## Local Architecture (Phase 0–2)

* **Inference & Hook Engine (Native Windows Python + RTX 5060 CUDA):** Intercepts transformer residual streams using PyTorch `register_forward_hook` and caches active steering vectors in GPU VRAM.
* **Redis:** Ephemeral hot-path inference queue and policy version cache/invalidation.
* **PostgreSQL (`pgvector`):** Relational system of record for calibration lineage, versioned steering policies, audit telemetry, and async drift clustering.
* **Prometheus & Grafana:** Live latency/intervention metrics (`host.docker.internal:8000/metrics`) and SQL-backed drift analytics.

## Quickstart (Phase 0)

1. Copy `.env.example` to `.env`.
2. Start Docker Desktop, then launch backing infrastructure:
   ```powershell
   docker compose up -d
   ```
3. Sync Python dependencies via `uv`:
   ```powershell
   uv sync
   ```
