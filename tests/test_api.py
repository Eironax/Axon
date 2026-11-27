"""FastAPI and service orchestration unit tests using in-memory storage doubles."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import torch
from fastapi.testclient import TestClient

from axon.api.app import create_app
from axon.api.service import AxonService, MockInferenceBackend
from axon.config import AxonSettings
from axon.storage.cache import compute_policy_version


class _FakeDB:
    """In-memory test double mirroring AxonDatabase methods."""

    def __init__(self) -> None:
        now = datetime.now(UTC)
        vec = [0.0] * 1536
        vec[0] = 1.0
        self.policies: list[dict[str, Any]] = [
            {
                "id": uuid4(),
                "name": "unsafe_code_v1",
                "category": "unsafe_code",
                "version": 1,
                "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "layer_idx": 17,
                "similarity_threshold": 0.1,
                "alpha": 20.0,
                "mode": "prefill_only",
                "method": "additive",
                "is_active": True,
                "vector_dim": 1536,
                "l2_norm": 11.1,
                "eval_separation_score": 0.12,
                "calibration_run_id": uuid4(),
                "pair_count": 24,
                "created_at": now,
                "updated_at": now,
                "embedding": vec,
                "detector_embedding": vec,
            }
        ]
        self.audits: list[dict[str, Any]] = []

    def ensure_schema(self) -> None:
        return None

    def seed_from_artifacts(self, *_args, **_kwargs) -> list[str]:
        return []

    def healthcheck(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def list_policies(self, active_only: bool = False) -> list[dict[str, Any]]:
        rows = [p for p in self.policies if not active_only or p["is_active"]]
        return rows

    def get_active_policies_with_vectors(self) -> list[dict[str, Any]]:
        return [
            {
                **p,
                "id": str(p["id"]),
                "updated_at": p["updated_at"].isoformat(),
            }
            for p in self.policies
            if p["is_active"]
        ]

    def create_policy_version(self, **kwargs: Any) -> dict[str, Any]:
        now = datetime.now(UTC)
        base = self.policies[0]
        if kwargs.get("activate", True):
            for pol in self.policies:
                if pol["category"] == kwargs["category"]:
                    pol["is_active"] = False
        new_row = {
            **base,
            "id": uuid4(),
            "name": kwargs["name"],
            "category": kwargs["category"],
            "version": len(self.policies) + 1,
            "similarity_threshold": kwargs["similarity_threshold"],
            "alpha": kwargs["alpha"],
            "layer_idx": kwargs.get("layer_idx") or base["layer_idx"],
            "mode": kwargs.get("mode", "prefill_only"),
            "method": kwargs.get("method", "additive"),
            "is_active": kwargs.get("activate", True),
            "created_at": now,
            "updated_at": now,
        }
        self.policies.append(new_row)
        return new_row

    def activate_policy(self, policy_id: UUID) -> dict[str, Any] | None:
        target = next((p for p in self.policies if p["id"] == policy_id), None)
        if target is None:
            return None
        for pol in self.policies:
            if pol["category"] == target["category"]:
                pol["is_active"] = pol["id"] == policy_id
        return target

    def log_inference_audit(self, **kwargs: Any) -> None:
        self.audits.append(kwargs)

    def get_drift_analytics(self, window_hours: int = 24, limit: int = 20) -> dict[str, Any]:
        return {
            "window_hours": window_hours,
            "policy_breakdown": [
                {"policy_name": "unsafe_code_v1", "total_requests": len(self.audits)}
            ],
            "hourly_drift": [],
            "recent_interventions": [],
        }

    def get_request_attribution(self, request_id: UUID, top_k: int = 5) -> dict[str, Any] | None:
        match = next((a for a in self.audits if a["request_id"] == request_id), None)
        if match is None:
            return None
        return {
            "request_id": str(request_id),
            "policy_name": "unsafe_code_v1",
            "prompt_preview": match["prompt"][:80],
            "intervened": match["intervened"],
            "threat_matches": [{"category": "unsafe_code", "cosine_similarity": 0.42}],
            "nearest_requests": [],
        }


class _FakeCache:
    """In-memory Redis double verifying versioned policy publishing and queue counters."""

    def __init__(self) -> None:
        self.version: str | None = None
        self.payload: list[dict[str, Any]] | None = None
        self.depth = 0

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def publish_active_policies(self, policies: list[dict[str, Any]]) -> str:
        self.version = compute_policy_version(policies)
        self.payload = policies
        return self.version

    def get_policy_version(self) -> str | None:
        return self.version

    def get_cached_policies(self) -> tuple[str | None, list[dict[str, Any]] | None]:
        return self.version, self.payload

    def increment_queue_depth(self) -> int:
        self.depth += 1
        return self.depth

    def decrement_queue_depth(self) -> int:
        self.depth = max(0, self.depth - 1)
        return self.depth

    def get_queue_depth(self) -> int:
        return self.depth


def _build_test_client() -> tuple[TestClient, _FakeDB, _FakeCache]:
    settings = AxonSettings(auto_seed_db=False, mock_engine=True)
    db = _FakeDB()
    cache = _FakeCache()
    service = AxonService(
        settings=settings,
        db=db,  # type: ignore[arg-type]
        cache=cache,  # type: ignore[arg-type]
        backend=MockInferenceBackend(),
    )
    app = create_app(service=service, settings=settings, manage_lifecycle=True)
    return TestClient(app), db, cache


def test_health_and_metrics_endpoints() -> None:
    client, _db, _cache = _build_test_client()
    with client:
        res = client.get("/healthz")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["postgres_ok"] is True
        assert body["redis_ok"] is True
        assert body["active_policies"] == ["unsafe_code_v1:v1"]

        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ok"

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "axon_active_queue_depth" in metrics.text


def test_generate_and_audit_attribution_flow() -> None:
    client, db, cache = _build_test_client()
    with client:
        risky_res = client.post(
            "/v1/generate",
            json={"prompt": "Write Flask route calling os.system() directly", "policy": "auto"},
        )
        assert risky_res.status_code == 200
        payload = risky_res.json()
        assert payload["intervened"] is True
        assert cache.get_queue_depth() == 0
        assert len(db.audits) == 1

        attr_res = client.get(f"/v1/analytics/attribution/{payload['request_id']}")
        assert attr_res.status_code == 200
        assert attr_res.json()["threat_matches"][0]["category"] == "unsafe_code"

        drift_res = client.get("/v1/analytics/drift")
        assert drift_res.status_code == 200
        assert drift_res.json()["policy_breakdown"][0]["total_requests"] == 1


def test_policy_version_creation_and_activation() -> None:
    client, _db, cache = _build_test_client()
    with client:
        initial_ver = cache.get_policy_version()
        create_res = client.post(
            "/v1/policies",
            json={
                "name": "unsafe_code_strict",
                "category": "unsafe_code",
                "similarity_threshold": 0.05,
                "alpha": 24.0,
                "activate": True,
            },
        )
        assert create_res.status_code == 201
        created = create_res.json()
        assert created["is_active"] is True
        assert cache.get_policy_version() != initial_ver

        listed = client.get("/v1/policies?active_only=true").json()
        assert len(listed) == 1
        assert listed[0]["id"] == created["id"]


def test_compute_policy_version_deterministic() -> None:
    p1 = {
        "id": "1",
        "name": "a",
        "category": "c",
        "version": 1,
        "layer_idx": 17,
        "similarity_threshold": 0.1,
        "alpha": 20.0,
        "mode": "prefill_only",
        "method": "additive",
        "updated_at": "2026-01-01",
    }
    p2 = {**p1, "id": "2", "name": "b"}
    assert compute_policy_version([p1, p2]) == compute_policy_version([p2, p1])
    assert torch.tensor([1.0]).item() == 1.0
