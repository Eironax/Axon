"""FastAPI application exposing latent firewall inference, policy control, and drift SQL."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import anyio
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from axon import __version__
from axon.api.schemas import (
    DriftAnalyticsResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    PolicyCreateRequest,
    PolicySummary,
    RequestAttributionResponse,
)
from axon.api.service import AxonService
from axon.config import AxonSettings, get_settings
from axon.storage.cache import AxonRedisCache
from axon.storage.db import AxonDatabase


def create_app(
    service: AxonService | None = None,
    settings: AxonSettings | None = None,
    manage_lifecycle: bool = True,
) -> FastAPI:
    """Create configured FastAPI app with optional injected service for testing."""
    cfg = settings or get_settings()
    if service is None:
        db = AxonDatabase(cfg.postgres_dsn)
        cache = AxonRedisCache(cfg.redis_url)
        svc = AxonService(settings=cfg, db=db, cache=cache)
    else:
        svc = service

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if manage_lifecycle:
            await anyio.to_thread.run_sync(svc.startup)
        try:
            yield
        finally:
            if manage_lifecycle:
                await anyio.to_thread.run_sync(svc.shutdown)

    application = FastAPI(
        title="Axon Latent Firewall",
        version=__version__,
        description="Runtime PyTorch activation-steering firewall and drift analytics service.",
        lifespan=lifespan,
    )
    application.state.axon_service = svc

    Instrumentator(excluded_handlers=["/metrics", "/healthz", "/readyz"]).instrument(
        application
    ).expose(
        application,
        endpoint="/metrics",
        include_in_schema=False,
    )

    @application.get("/healthz", response_model=HealthResponse, tags=["system"])
    async def healthz() -> HealthResponse:
        return await anyio.to_thread.run_sync(svc.health)

    @application.get("/readyz", response_model=HealthResponse, tags=["system"])
    async def readyz() -> JSONResponse:
        status = await anyio.to_thread.run_sync(svc.health)
        code = 200 if status.status == "ok" else 503
        return JSONResponse(status_code=code, content=status.model_dump(mode="json"))

    @application.post("/v1/generate", response_model=GenerateResponse, tags=["inference"])
    async def generate(req: GenerateRequest, background_tasks: BackgroundTasks) -> GenerateResponse:
        if svc.backend is None:
            raise HTTPException(status_code=503, detail="Inference backend is not initialized")
        try:
            response, result, policy_id = await svc.generate(req)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        background_tasks.add_task(svc.record_audit_sync, response.request_id, policy_id, result)
        return response

    @application.get("/v1/policies", response_model=list[PolicySummary], tags=["policies"])
    async def list_policies(active_only: bool = Query(default=False)) -> list[PolicySummary]:
        rows = await anyio.to_thread.run_sync(svc.db.list_policies, active_only)
        return [PolicySummary(**row) for row in rows]

    @application.post(
        "/v1/policies", response_model=PolicySummary, status_code=201, tags=["policies"]
    )
    async def create_policy(req: PolicyCreateRequest) -> PolicySummary:
        try:
            row = await anyio.to_thread.run_sync(svc.create_policy, req)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return PolicySummary(**row)

    @application.post(
        "/v1/policies/{policy_id}/activate",
        response_model=PolicySummary,
        tags=["policies"],
    )
    async def activate_policy(policy_id: UUID) -> PolicySummary:
        row = await anyio.to_thread.run_sync(svc.activate_policy, policy_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
        return PolicySummary(**row)

    @application.get(
        "/v1/analytics/drift",
        response_model=DriftAnalyticsResponse,
        tags=["analytics"],
    )
    async def drift_analytics(
        window_hours: int = Query(default=24, ge=1, le=720),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> DriftAnalyticsResponse:
        payload = await anyio.to_thread.run_sync(svc.db.get_drift_analytics, window_hours, limit)
        return DriftAnalyticsResponse(**payload)

    @application.get(
        "/v1/analytics/attribution/{request_id}",
        response_model=RequestAttributionResponse,
        tags=["analytics"],
    )
    async def request_attribution(
        request_id: UUID,
        top_k: int = Query(default=5, ge=1, le=20),
    ) -> RequestAttributionResponse:
        payload = await anyio.to_thread.run_sync(svc.db.get_request_attribution, request_id, top_k)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found")
        return RequestAttributionResponse(**payload)

    return application


app = create_app()
