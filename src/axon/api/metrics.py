"""Prometheus metrics exported by the Axon inference firewall."""

from prometheus_client import Counter, Gauge, Histogram

HOOK_LATENCY_MS = Histogram(
    "axon_hook_latency_ms",
    "Per-forward-pass activation hook latency in milliseconds",
    labelnames=("policy", "category"),
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 10.0),
)

TTFT_MS = Histogram(
    "axon_ttft_ms",
    "Time-To-First-Token in milliseconds",
    labelnames=("policy",),
    buckets=(5.0, 15.0, 25.0, 35.0, 50.0, 75.0, 100.0, 200.0, 500.0, 1000.0),
)

TOKENS_PER_SEC = Histogram(
    "axon_tokens_per_sec",
    "Generated token throughput per request",
    labelnames=("policy",),
    buckets=(5.0, 15.0, 25.0, 35.0, 45.0, 60.0, 80.0, 120.0),
)

COSINE_SIMILARITY = Histogram(
    "axon_cosine_similarity",
    "Observed cosine similarity against active detector/steering vectors",
    labelnames=("policy", "phase"),
    buckets=(-0.6, -0.4, -0.2, -0.05, 0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8),
)

INTERVENTIONS_TOTAL = Counter(
    "axon_interventions_total",
    "Total number of generations where activation steering intervened",
    labelnames=("policy", "category"),
)

REQUESTS_TOTAL = Counter(
    "axon_requests_total",
    "Total firewall inference requests processed",
    labelnames=("policy", "intervened"),
)

ACTIVE_QUEUE_DEPTH = Gauge(
    "axon_active_queue_depth",
    "Current number of queued and in-flight inference requests",
)

VRAM_ALLOCATED_BYTES = Gauge(
    "axon_vram_allocated_bytes",
    "CUDA memory currently allocated by the inference process in bytes",
)
