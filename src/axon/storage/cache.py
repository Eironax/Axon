"""Redis cache for active policy versions, vectors, pub/sub updates, and queue depth."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

ACTIVE_POLICIES_KEY = "axon:policies:active"
POLICY_VERSION_KEY = "axon:policies:version"
POLICY_UPDATES_CHANNEL = "axon:policy:updates"
QUEUE_DEPTH_KEY = "axon:queue:depth"


def compute_policy_version(policies: list[dict[str, Any]]) -> str:
    """Return deterministic version hash for active steering policy configs."""
    fingerprint = [
        {
            "id": str(p["id"]),
            "name": p["name"],
            "category": p["category"],
            "version": p["version"],
            "layer_idx": p["layer_idx"],
            "similarity_threshold": p["similarity_threshold"],
            "alpha": p["alpha"],
            "mode": p["mode"],
            "method": p["method"],
            "updated_at": str(p["updated_at"]),
        }
        for p in sorted(policies, key=lambda item: (item["category"], item["name"]))
    ]
    encoded = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class AxonRedisCache:
    """Thin Redis wrapper managing policy caching, invalidation pub/sub, and queue depth."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        """Check Redis liveness."""
        return bool(self.client.ping())

    def close(self) -> None:
        """Close underlying Redis connection pool."""
        self.client.close()

    def publish_active_policies(self, policies: list[dict[str, Any]]) -> str:
        """Cache serialized active policies and broadcast invalidation version."""
        version = compute_policy_version(policies)
        payload = json.dumps(policies)
        pipe = self.client.pipeline()
        pipe.set(ACTIVE_POLICIES_KEY, payload)
        pipe.set(POLICY_VERSION_KEY, version)
        pipe.publish(POLICY_UPDATES_CHANNEL, version)
        pipe.execute()
        return version

    def get_policy_version(self) -> str | None:
        """Read current active policy version hash without pulling vector payloads."""
        return self.client.get(POLICY_VERSION_KEY)

    def get_cached_policies(self) -> tuple[str | None, list[dict[str, Any]] | None]:
        """Fetch cached version hash and active policy vectors."""
        pipe = self.client.pipeline()
        pipe.get(POLICY_VERSION_KEY)
        pipe.get(ACTIVE_POLICIES_KEY)
        version, raw = pipe.execute()
        if not raw:
            return version, None
        return version, json.loads(raw)

    def increment_queue_depth(self) -> int:
        """Atomically increment active inference queue depth."""
        return int(self.client.incr(QUEUE_DEPTH_KEY))

    def decrement_queue_depth(self) -> int:
        """Atomically decrement queue depth, clamping safely at zero."""
        value = int(self.client.decr(QUEUE_DEPTH_KEY))
        if value < 0:
            self.client.set(QUEUE_DEPTH_KEY, 0)
            return 0
        return value

    def get_queue_depth(self) -> int:
        """Read current queue depth."""
        raw = self.client.get(QUEUE_DEPTH_KEY)
        return max(0, int(raw)) if raw is not None else 0
