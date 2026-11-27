"""Service orchestration connecting inference, Redis policy cache, Postgres, and metrics."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import anyio
import torch

from axon.api.metrics import (
    ACTIVE_QUEUE_DEPTH,
    COSINE_SIMILARITY,
    HOOK_LATENCY_MS,
    INTERVENTIONS_TOTAL,
    REQUESTS_TOTAL,
    TOKENS_PER_SEC,
    TTFT_MS,
    VRAM_ALLOCATED_BYTES,
)
from axon.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    PolicyCreateRequest,
)
from axon.config import AxonSettings
from axon.engine.datasets import POLICY_DATASETS
from axon.engine.firewall import (
    FirewallResult,
    LatentFirewall,
    SteeringMethod,
    SteeringMode,
    SteeringPolicy,
)
from axon.engine.model import load_model_and_tokenizer
from axon.storage.cache import AxonRedisCache
from axon.storage.db import EXPECTED_VECTOR_DIM, AxonDatabase

logger = logging.getLogger(__name__)


class InferenceBackend(Protocol):
    """Protocol implemented by both real CUDA LatentFirewall and CI mock backend."""

    device_label: str

    def select_policy(
        self,
        prompt: str,
        policies: list[SteeringPolicy],
        system_prompt: str | None = None,
    ) -> SteeringPolicy | None: ...

    def generate(
        self,
        prompt: str,
        policy: SteeringPolicy | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        capture_prefill_vector: bool = True,
        sync_hook_timing: bool = False,
    ) -> FirewallResult: ...


class RealInferenceBackend:
    """Live HuggingFace + PyTorch forward-hook backend."""

    def __init__(self, settings: AxonSettings) -> None:
        loaded = load_model_and_tokenizer(
            settings.model_id,
            dtype_name=settings.torch_dtype,
            device_name=settings.device,
        )
        self.device_label = str(loaded.device)
        self.firewall = LatentFirewall(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            device=loaded.device,
            dtype=loaded.dtype,
        )
        # Warm up CUDA kernels during startup so first live request avoids JIT latency
        self.firewall.generate(
            prompt="warmup",
            policy=None,
            max_new_tokens=2,
            capture_prefill_vector=False,
        )

    def select_policy(
        self,
        prompt: str,
        policies: list[SteeringPolicy],
        system_prompt: str | None = None,
    ) -> SteeringPolicy | None:
        return self.firewall.select_policy(prompt, policies, system_prompt=system_prompt)

    def generate(
        self,
        prompt: str,
        policy: SteeringPolicy | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        capture_prefill_vector: bool = True,
        sync_hook_timing: bool = False,
    ) -> FirewallResult:
        return self.firewall.generate(
            prompt=prompt,
            policy=policy,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            capture_prefill_vector=capture_prefill_vector,
            sync_hook_timing=sync_hook_timing,
        )


class MockInferenceBackend:
    """Lightweight deterministic backend used only when AXON_MOCK_ENGINE=true."""

    device_label = "mock-cpu"

    _RISK_KEYWORDS = (
        "ignore",
        "override",
        "system prompt",
        "os.system",
        "shell=true",
        "eval(",
        "pickle.loads",
        "hardcoded",
        "raw sql",
        "strcpy",
    )

    def _score_prompt(self, prompt: str, policy: SteeringPolicy) -> float:
        lower = prompt.lower()
        hits = sum(1 for kw in self._RISK_KEYWORDS if kw in lower)
        if hits > 0:
            return min(0.85, max(policy.similarity_threshold + 0.12, 0.25))
        return -0.18

    def select_policy(
        self,
        prompt: str,
        policies: list[SteeringPolicy],
        system_prompt: str | None = None,
    ) -> SteeringPolicy | None:
        if not policies:
            return None
        lower = prompt.lower()
        if any(tok in lower for tok in ("ignore", "override", "system prompt")):
            for pol in policies:
                if pol.category == "prompt_injection":
                    return pol
        return max(policies, key=lambda p: self._score_prompt(prompt, p) - p.similarity_threshold)

    def generate(
        self,
        prompt: str,
        policy: SteeringPolicy | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        capture_prefill_vector: bool = True,
        sync_hook_timing: bool = False,
    ) -> FirewallResult:
        time.sleep(0.02)
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        base = torch.tensor(
            [((b / 255.0) - 0.5) for b in digest] * (EXPECTED_VECTOR_DIM // len(digest)),
            dtype=torch.float32,
        )
        prefill_sim = 0.0
        intervened = False
        if policy is not None:
            prefill_sim = self._score_prompt(prompt, policy)
            intervened = prefill_sim >= policy.similarity_threshold or policy.mode == "always"
            if intervened:
                det = policy.normalized_detector_vector()
                if det.numel() == EXPECTED_VECTOR_DIM:
                    base = torch.nn.functional.normalize(base + 0.6 * det, p=2, dim=0)

        text = (
            "Request steered away from unsafe pattern by Axon mock firewall."
            if intervened
            else f"Safe mock response for: {prompt[:80]}"
        )
        tokens = max(8, min(max_new_tokens, len(text.split()) * 2))
        return FirewallResult(
            prompt=prompt,
            response_text=text,
            policy_name=policy.name if policy is not None else "baseline_unsteered",
            prefill_cosine_sim=prefill_sim,
            peak_decode_sim=prefill_sim * 0.5,
            intervened=intervened,
            intervention_tokens=tokens if intervened else 0,
            generated_tokens=tokens,
            ttft_ms=18.5,
            total_generation_ms=42.0,
            tokens_per_sec=tokens / 0.042,
            hook_avg_ms=0.22,
            hook_p99_ms=0.35,
            prefill_activation=base.tolist() if capture_prefill_vector else None,
        )


@dataclass(slots=True)
class ActivePolicyRecord:
    """In-memory active policy pairing DB UUID with runtime SteeringPolicy."""

    policy_id: UUID
    name: str
    category: str
    version: int
    policy: SteeringPolicy


def _build_policy_record(row: dict[str, Any]) -> ActivePolicyRecord:
    mode: SteeringMode = row["mode"]
    method: SteeringMethod = row["method"]
    steering_policy = SteeringPolicy(
        name=row["name"],
        category=row["category"],
        layer_idx=int(row["layer_idx"]),
        vector=torch.tensor(row["embedding"], dtype=torch.float32),
        detector_vector=torch.tensor(row["detector_embedding"], dtype=torch.float32),
        similarity_threshold=float(row["similarity_threshold"]),
        alpha=float(row["alpha"]),
        mode=mode,
        method=method,
    )
    return ActivePolicyRecord(
        policy_id=UUID(str(row["id"])),
        name=row["name"],
        category=row["category"],
        version=int(row["version"]),
        policy=steering_policy,
    )


class AxonService:
    """Core service coordinating Postgres, Redis, GPU inference, and telemetry."""

    def __init__(
        self,
        settings: AxonSettings,
        db: AxonDatabase,
        cache: AxonRedisCache,
        backend: InferenceBackend | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.cache = cache
        self.backend = backend
        self._policies_lock = threading.Lock()
        self._active_policies: list[ActivePolicyRecord] = []
        self._policy_version: str | None = None
        self._inference_semaphore = threading.Semaphore(max(1, settings.max_concurrent_generations))

    def startup(self) -> None:
        """Initialize schema, seed artifacts, sync Redis policy cache, and load model."""
        for attempt in range(1, 16):
            try:
                self.db.ensure_schema()
                if self.settings.auto_seed_db:
                    seeded = self.db.seed_from_artifacts(
                        Path(self.settings.artifacts_dir), POLICY_DATASETS
                    )
                    if seeded:
                        logger.info("Seeded initial steering policies: %s", seeded)
                self.refresh_policy_cache()
                break
            except Exception as exc:
                if attempt == 15:
                    logger.exception(
                        "Failed to initialize database/Redis policy cache after %d attempts",
                        attempt,
                    )
                else:
                    logger.warning(
                        "Waiting for Postgres/Redis readiness (attempt %d/15): %s",
                        attempt,
                        exc,
                    )
                    time.sleep(2.0)

        if self.backend is None:
            if self.settings.mock_engine:
                self.backend = MockInferenceBackend()
            else:
                self.backend = RealInferenceBackend(self.settings)

        if torch.cuda.is_available():
            VRAM_ALLOCATED_BYTES.set(float(torch.cuda.memory_allocated()))

    def shutdown(self) -> None:
        """Release database and Redis connections cleanly."""
        self.cache.close()
        self.db.close()

    def refresh_policy_cache(self) -> str:
        """Reload active policies from PostgreSQL, publish to Redis, and stage locally."""
        rows = self.db.get_active_policies_with_vectors()
        version = self.cache.publish_active_policies(rows)
        records = [_build_policy_record(row) for row in rows]
        with self._policies_lock:
            self._active_policies = records
            self._policy_version = version
        return version

    def _ensure_fresh_policies(self) -> list[ActivePolicyRecord]:
        try:
            remote_ver = self.cache.get_policy_version()
        except Exception:
            remote_ver = None

        with self._policies_lock:
            current_ver = self._policy_version
            current_policies = list(self._active_policies)

        if remote_ver is not None and remote_ver != current_ver:
            _, cached_rows = self.cache.get_cached_policies()
            if cached_rows is not None:
                records = [_build_policy_record(row) for row in cached_rows]
                with self._policies_lock:
                    self._active_policies = records
                    self._policy_version = remote_ver
                    return list(records)
            self.refresh_policy_cache()
            with self._policies_lock:
                return list(self._active_policies)
        return current_policies

    def _resolve_policy_record(
        self,
        req: GenerateRequest,
        active_records: list[ActivePolicyRecord],
    ) -> ActivePolicyRecord | None:
        assert self.backend is not None
        if req.policy == "none":
            return None
        if req.policy == "auto":
            if not active_records:
                return None
            chosen = self.backend.select_policy(
                req.prompt,
                [rec.policy for rec in active_records],
                system_prompt=req.system_prompt,
            )
            if chosen is None:
                return None
            for rec in active_records:
                if rec.name == chosen.name:
                    return rec
            return active_records[0]

        for rec in active_records:
            if req.policy in (rec.category, rec.name):
                return rec
        raise KeyError(f"No active steering policy matching '{req.policy}'")

    def _run_generation_sync(
        self, req: GenerateRequest
    ) -> tuple[UUID, ActivePolicyRecord | None, FirewallResult]:
        assert self.backend is not None
        request_id = uuid4()
        active_records = self._ensure_fresh_policies()
        selected = self._resolve_policy_record(req, active_records)

        runtime_policy: SteeringPolicy | None = None
        if selected is not None:
            base = selected.policy
            runtime_policy = SteeringPolicy(
                name=base.name,
                category=base.category,
                layer_idx=base.layer_idx,
                vector=base.vector,
                detector_vector=base.detector_vector,
                similarity_threshold=(
                    req.threshold if req.threshold is not None else base.similarity_threshold
                ),
                alpha=req.alpha if req.alpha is not None else base.alpha,
                mode=base.mode,
                method=base.method,
            )

        depth = self.cache.increment_queue_depth()
        ACTIVE_QUEUE_DEPTH.set(float(depth))
        try:
            with self._inference_semaphore:
                result = self.backend.generate(
                    prompt=req.prompt,
                    policy=runtime_policy,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    system_prompt=req.system_prompt,
                    capture_prefill_vector=True,
                    sync_hook_timing=False,
                )
        finally:
            remaining = self.cache.decrement_queue_depth()
            ACTIVE_QUEUE_DEPTH.set(float(remaining))

        category = selected.category if selected is not None else "none"
        HOOK_LATENCY_MS.labels(policy=result.policy_name, category=category).observe(
            result.hook_avg_ms
        )
        TTFT_MS.labels(policy=result.policy_name).observe(result.ttft_ms)
        TOKENS_PER_SEC.labels(policy=result.policy_name).observe(result.tokens_per_sec)
        COSINE_SIMILARITY.labels(policy=result.policy_name, phase="prefill").observe(
            result.prefill_cosine_sim
        )
        COSINE_SIMILARITY.labels(policy=result.policy_name, phase="decode_peak").observe(
            result.peak_decode_sim
        )
        REQUESTS_TOTAL.labels(
            policy=result.policy_name, intervened=str(result.intervened).lower()
        ).inc()
        if result.intervened:
            INTERVENTIONS_TOTAL.labels(policy=result.policy_name, category=category).inc()
        if torch.cuda.is_available():
            VRAM_ALLOCATED_BYTES.set(float(torch.cuda.memory_allocated()))

        return request_id, selected, result

    async def generate(
        self, req: GenerateRequest
    ) -> tuple[GenerateResponse, FirewallResult, UUID | None]:
        """Run steered generation off the asyncio loop and return structured response."""
        request_id, selected, result = await anyio.to_thread.run_sync(
            self._run_generation_sync, req
        )
        response = GenerateResponse(
            request_id=request_id,
            policy_id=selected.policy_id if selected is not None else None,
            policy_name=result.policy_name,
            category=selected.category if selected is not None else "none",
            response_text=result.response_text,
            prefill_cosine_sim=result.prefill_cosine_sim,
            peak_decode_sim=result.peak_decode_sim,
            intervened=result.intervened,
            intervention_tokens=result.intervention_tokens,
            generated_tokens=result.generated_tokens,
            ttft_ms=result.ttft_ms,
            total_generation_ms=result.total_generation_ms,
            tokens_per_sec=result.tokens_per_sec,
            hook_avg_ms=result.hook_avg_ms,
            hook_p99_ms=result.hook_p99_ms,
        )
        return response, result, (selected.policy_id if selected is not None else None)

    def record_audit_sync(
        self,
        request_id: UUID,
        policy_id: UUID | None,
        result: FirewallResult,
    ) -> None:
        """Persist post-response audit row and prefill activation into pgvector."""
        try:
            self.db.log_inference_audit(
                request_id=request_id,
                policy_id=policy_id,
                model_id=self.settings.model_id,
                prompt=result.prompt,
                prefill_cosine_sim=result.prefill_cosine_sim,
                peak_decode_sim=result.peak_decode_sim,
                intervened=result.intervened,
                intervention_tokens=result.intervention_tokens,
                generated_tokens=result.generated_tokens,
                ttft_ms=result.ttft_ms,
                hook_avg_ms=result.hook_avg_ms,
                hook_p99_ms=result.hook_p99_ms,
                prefill_activation=result.prefill_activation,
            )
        except Exception:
            logger.exception("Failed writing audit record for request_id=%s", request_id)

    def create_policy(self, req: PolicyCreateRequest) -> dict[str, Any]:
        """Create a new policy version in Postgres and refresh Redis cache."""
        created = self.db.create_policy_version(
            name=req.name,
            category=req.category,
            similarity_threshold=req.similarity_threshold,
            alpha=req.alpha,
            layer_idx=req.layer_idx,
            mode=req.mode,
            method=req.method,
            activate=req.activate,
        )
        self.refresh_policy_cache()
        rows = self.db.list_policies(active_only=False)
        for row in rows:
            if row["id"] == created["id"]:
                return row
        return created

    def activate_policy(self, policy_id: UUID) -> dict[str, Any] | None:
        """Activate a specific policy version and invalidate worker cache via Redis."""
        updated = self.db.activate_policy(policy_id)
        if updated is None:
            return None
        self.refresh_policy_cache()
        rows = self.db.list_policies(active_only=False)
        for row in rows:
            if row["id"] == policy_id:
                return row
        return updated

    def health(self) -> HealthResponse:
        """Check Postgres, Redis, and inference engine status."""
        try:
            pg_ok = self.db.healthcheck()
        except Exception:
            pg_ok = False
        try:
            redis_ok = self.cache.ping()
            depth = self.cache.get_queue_depth()
        except Exception:
            redis_ok = False
            depth = 0

        with self._policies_lock:
            active_names = [f"{p.name}:v{p.version}" for p in self._active_policies]

        model_loaded = self.backend is not None
        status = "ok" if (pg_ok and redis_ok and model_loaded) else "degraded"
        return HealthResponse(
            status=status,
            postgres_ok=pg_ok,
            redis_ok=redis_ok,
            model_loaded=model_loaded,
            device=self.backend.device_label if self.backend is not None else "unloaded",
            queue_depth=depth,
            active_policies=active_names,
        )
