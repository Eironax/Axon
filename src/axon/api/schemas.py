"""Pydantic request and response models for Axon REST endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Input payload for latent firewall inference."""

    prompt: str = Field(..., min_length=1, max_length=16000)
    policy: str = Field(
        default="auto",
        description="Policy category/name, 'auto' to route by margin, or 'none' for baseline.",
    )
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    alpha: float | None = Field(default=None, ge=0.0, le=100.0)
    threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    system_prompt: str | None = Field(default=None, max_length=4000)


class GenerateResponse(BaseModel):
    """Generated text accompanied by hook steering and latency metrics."""

    request_id: UUID
    policy_id: UUID | None
    policy_name: str
    category: str
    response_text: str
    prefill_cosine_sim: float
    peak_decode_sim: float
    intervened: bool
    intervention_tokens: int
    generated_tokens: int
    ttft_ms: float
    total_generation_ms: float
    tokens_per_sec: float
    hook_avg_ms: float
    hook_p99_ms: float


class PolicyCreateRequest(BaseModel):
    """Request to create a new versioned steering policy inheriting calibrated vectors."""

    name: str = Field(..., min_length=2, max_length=120)
    category: str = Field(..., min_length=2, max_length=64)
    similarity_threshold: float = Field(..., ge=-1.0, le=1.0)
    alpha: float = Field(..., ge=0.0, le=100.0)
    layer_idx: int | None = Field(default=None, ge=0, le=128)
    mode: Literal["hybrid", "prefill_only", "decode_only", "always"] = "prefill_only"
    method: Literal["additive", "projection"] = "additive"
    activate: bool = True


class PolicySummary(BaseModel):
    """Policy record joined with vector and calibration lineage."""

    id: UUID
    name: str
    category: str
    version: int
    model_id: str
    layer_idx: int
    similarity_threshold: float
    alpha: float
    mode: str
    method: str
    is_active: bool
    vector_dim: int
    l2_norm: float
    eval_separation_score: float | None
    calibration_run_id: UUID
    pair_count: int
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    """Service readiness and dependency probe status."""

    status: Literal["ok", "degraded"]
    postgres_ok: bool
    redis_ok: bool
    model_loaded: bool
    device: str
    queue_depth: int
    active_policies: list[str]


class DriftAnalyticsResponse(BaseModel):
    """SQL and time-windowed drift telemetry payload."""

    window_hours: int
    policy_breakdown: list[dict[str, Any]]
    hourly_drift: list[dict[str, Any]]
    recent_interventions: list[dict[str, Any]]


class RequestAttributionResponse(BaseModel):
    """pgvector nearest-neighbor threat attribution for an audited request."""

    request_id: str
    policy_name: str | None
    prompt_preview: str
    intervened: bool
    threat_matches: list[dict[str, Any]]
    nearest_requests: list[dict[str, Any]]
