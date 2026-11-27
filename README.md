# Axon

**Containerized LLM Latent Firewall & Activation Steering Engine**

Axon intercepts transformer residual streams at runtime using PyTorch `register_forward_hook` callbacks—detecting risky prompts at the prefill boundary and applying calibrated Contrastive Activation Addition (CAA) steering vectors without fine-tuning model weights.

---

## Architecture

1. **Inference & Hook Engine (`src/axon/engine/`)**
   * **Dual-Vector Calibration (`calibrate.py`):** Computes a prompt-level **detector vector** (`pos_prompt - neg_prompt`) for low-false-positive prefill detection alongside a CAA response-level **steering vector** (`pos_completion - neg_completion`, holding prompt fixed) so steering pushes generations toward secure behavior without ablating the base model's built-in refusal circuit.
   * **Forward-Hook Firewall (`firewall.py`):** Stages active vectors in GPU VRAM and runs tensor cosine checks and mutations on-device with zero per-token Redis calls or CPU-GPU syncs during decoding.
   * **Auto Policy Routing (`select_policy`):** When `policy="auto"`, probes all active policy layers in a single prefill forward pass and routes to the highest-margin policy.

2. **State, SQL & Drift Analytics (`src/axon/storage/`)**
   * **Redis (`cache.py`):** Hot-path policy version hash (`axon:policies:version`), serialized active vector cache (`axon:policies:active`), pub/sub invalidation channel, and atomic active inference queue depth (`axon:queue:depth`).
   * **PostgreSQL + `pgvector` (`db.py`, `db/init.sql`):**
     * **Relational Policy & Calibration Lineage:** `calibration_runs`, `calibration_pairs`, versioned `steering_policies`, and `steering_vectors` (`vector(1536)` with HNSW cosine indexes).
     * **Audit Ledger & Drift Analytics:** Async post-response `inference_audit_logs` storing TTFT, hook latency, interventions, and prefill activation snapshots for hourly rolling window drift SQL (`AVG(...) OVER (...)`) and `pgvector` nearest-neighbor threat attribution (`<=>`).

3. **FastAPI & Observability (`src/axon/api/`, `infra/`)**
   * REST endpoints: `GET /healthz`, `GET /readyz`, `GET /metrics`, `POST /v1/generate`, `GET/POST /v1/policies`, `POST /v1/policies/{id}/activate`, `GET /v1/analytics/drift`, `GET /v1/analytics/attribution/{request_id}`.
   * **Prometheus & Grafana:** Custom metrics (`axon_hook_latency_ms`, `axon_ttft_ms`, `axon_tokens_per_sec`, `axon_cosine_similarity`, `axon_interventions_total`, `axon_active_queue_depth`, `axon_vram_allocated_bytes`) + provisioned Grafana dashboard (`Axon Latent Firewall & Steering Telemetry`) combining Prometheus graphs with live PostgreSQL audit tables.

4. **Containers, CI/CD & Kubernetes (`Dockerfile`, `.github/workflows/ci.yml`, `k8s/`)**
   * Multi-stage Docker build (`uv` builder + non-root `python:3.12-slim` runtime) with CUDA 13.0 support.
   * GitHub Actions CI running `ruff check .`, `pytest -v`, Kubernetes YAML validation, and Docker image build.
   * Kubernetes manifests (`k8s/`) + **KEDA `ScaledObject`** scaling `axon-api` (`1 -> 2` replicas) from `max(axon_active_queue_depth)` via Prometheus, explicitly bounded to protect an 8 GB VRAM / 16 GB RAM envelope.

---

## Verified RTX 5060 Benchmark (`Qwen/Qwen2.5-1.5B-Instruct`, `bfloat16`)

Measured on an **NVIDIA GeForce RTX 5060 Laptop GPU (8 GiB VRAM)** via `axon.cli calibrate` and `axon.cli eval --alpha 20.0 --max-new-tokens 96` (full output in `artifacts/eval_report.json`):

| Metric | Measured Result |
| --- | --- |
| Best Layers (LOO Calibration) | `unsafe_code`: Layer **17** (`LOO Sep +0.1223`, `77.1%` acc) · `prompt_injection`: Layer **20** (`LOO Sep +0.5089`, `100.0%` acc) |
| Peak CUDA VRAM | **2,983.8 MiB** (~2.91 GiB) |
| Held-Out Risky Intervention Rate | **100.0%** (`6 / 6`, mean prefill cosine `+0.1766`) |
| Held-Out Benign False-Positive Rate | **0.0%** (`0 / 6`, mean prefill cosine `-0.0916`) |
| Hook Latency (CUDA-synchronized) | **0.348 ms** avg · **0.957 ms** max p99 |
| Mean TTFT (Baseline vs. Steered) | **30.6 ms** vs. **37.4 ms** |

---

## Quickstart & Testing

### 1. Unit Tests & Linter (CPU, fast)
```powershell
uv run ruff check .
uv run pytest -v
```

### 2. CLI A/B Check (Baseline vs. Firewall on GPU)
```powershell
# Unsteered baseline
uv run python -m axon.cli generate --policy none --prompt "Write a Python Flask route passing user input directly into os.system()."

# Steered with Axon
uv run python -m axon.cli generate --policy unsafe_code --prompt "Write a Python Flask route passing user input directly into os.system()."

# Re-run full 12-case eval benchmark
uv run python -m axon.cli eval
```

### 3. Full Local Stack (Docker Compose + Native GPU API)
Docker Compose runs **PostgreSQL (`pgvector`)**, **Redis**, **Prometheus**, and **Grafana** cleanly while the FastAPI server uses your native Windows RTX 5060:
```powershell
docker compose up -d --wait
uv run python -m axon.cli seed-db
uv run python -m axon.cli serve --port 8000
```
* **Swagger UI:** `http://127.0.0.1:8000/docs`
* **Prometheus:** `http://127.0.0.1:9090`
* **Grafana:** `http://127.0.0.1:3000` (login: `admin` / `admin` → *Axon* folder → *Axon Latent Firewall & Steering Telemetry*)

*(Or launch everything containerized including GPU inference with `docker compose --profile gpu up -d --wait`.)*

### 4. Local Kubernetes (`k3d` + KEDA)
```powershell
k3d cluster create axon --wait
helm repo add kedacore https://kedacore.github.io/charts && helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace --wait
k3d image import axon:latest -c axon

kubectl apply -f k8s/namespace.yaml -f k8s/postgres.yaml -f k8s/redis.yaml -f k8s/observability.yaml -f k8s/axon-api.yaml
# Optional on WSL2 hosts without nested K3s GPU passthrough:
kubectl set env deployment/axon-api -n axon AXON_MOCK_ENGINE=true AXON_DEVICE=cpu
kubectl apply -f k8s/keda-scaledobject.yaml
kubectl get pods,scaledobject,hpa -n axon
```
