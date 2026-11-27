"""PostgreSQL + pgvector repository for policy lineage, audit logs, and drift analytics."""

from __future__ import annotations

import hashlib
import queue
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from axon.engine.calibrate import load_calibration_artifact
from axon.engine.datasets import ContrastivePair

EXPECTED_VECTOR_DIM = 1536


def _to_vector_arg(values: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != EXPECTED_VECTOR_DIM:
        return None
    return arr


def _vector_to_list(value: Any) -> list[float]:
    if hasattr(value, "to_list"):
        return [float(x) for x in value.to_list()]
    return np.asarray(value, dtype=np.float32).tolist()


def _prompt_digest_and_preview(prompt: str, max_len: int = 180) -> tuple[str, str]:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    collapsed = " ".join(prompt.split())
    preview = collapsed if len(collapsed) <= max_len else collapsed[: max_len - 3] + "..."
    return digest, preview


class AxonDatabase:
    """Thread-safe PostgreSQL + pgvector client backed by a lightweight connection queue."""

    def __init__(self, dsn: str, pool_size: int = 4) -> None:
        self.dsn = dsn
        self._pool: queue.LifoQueue[psycopg.Connection[Any]] = queue.LifoQueue(maxsize=pool_size)
        self._pool_size = pool_size

    def _open_connection(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=False)
        register_vector(conn)
        return conn

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection[Any]]:
        """Borrow a registered pgvector connection and return it cleanly to the pool."""
        conn: psycopg.Connection[Any] | None = None
        try:
            conn = self._pool.get_nowait()
            if conn.closed:
                conn = self._open_connection()
        except queue.Empty:
            conn = self._open_connection()

        try:
            yield conn
        except Exception:
            if not conn.closed:
                conn.rollback()
            raise
        else:
            if not conn.closed:
                try:
                    self._pool.put_nowait(conn)
                except queue.Full:
                    conn.close()

    def close(self) -> None:
        """Close all pooled connections."""
        while True:
            try:
                conn = self._pool.get_nowait()
                if not conn.closed:
                    conn.close()
            except queue.Empty:
                break

    def healthcheck(self) -> bool:
        """Verify PostgreSQL connectivity and pgvector table availability."""
        with self.connection() as conn:
            row = conn.execute("SELECT 1 AS ok").fetchone()
            conn.commit()
            return bool(row and row["ok"] == 1)

    def ensure_schema(self, schema_sql_path: Path | None = None) -> None:
        """Run idempotent schema initialization before registering pgvector types."""
        if schema_sql_path is None:
            schema_sql_path = Path(__file__).resolve().parents[3] / "db" / "init.sql"
        if not schema_sql_path.exists():
            return
        sql = schema_sql_path.read_text(encoding="utf-8")
        # Open raw connection without register_vector() because CREATE EXTENSION vector
        # inside init.sql may not have executed yet on a fresh Postgres database.
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(sql)

    def seed_from_artifacts(
        self,
        artifacts_dir: Path,
        datasets: dict[str, tuple[ContrastivePair, ...]],
        default_alpha: float = 20.0,
    ) -> list[str]:
        """Seed initial calibration lineage, policies, and vectors if missing."""
        seeded: list[str] = []
        if not artifacts_dir.exists():
            return seeded

        for category, pairs in datasets.items():
            tensor_file = artifacts_dir / f"{category}.safetensors"
            meta_file = artifacts_dir / f"{category}.json"
            if not (tensor_file.exists() and meta_file.exists()):
                continue

            artifact = load_calibration_artifact(artifacts_dir, category)
            policy_name = f"{category}_v1"

            with self.connection() as conn:
                existing = conn.execute(
                    "SELECT id FROM steering_policies WHERE name = %s AND version = 1",
                    (policy_name,),
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    continue

                best_score = next(
                    (
                        s.loo_separation
                        for s in artifact.layer_scores
                        if s.layer_idx == artifact.best_layer
                    ),
                    0.0,
                )
                run_row = conn.execute(
                    """
                    INSERT INTO calibration_runs (model_id, layer_idx, pair_count, notes)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        artifact.model_id,
                        artifact.best_layer,
                        len(pairs),
                        f"Seeded Phase 1 calibration for {category}",
                    ),
                ).fetchone()
                assert run_row is not None
                run_id = run_row["id"]

                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO calibration_pairs (
                            run_id, category, positive_prompt, negative_prompt
                        )
                        VALUES (%s, %s, %s, %s)
                        """,
                        [(run_id, p.category, p.positive_prompt, p.negative_prompt) for p in pairs],
                    )

                active_row = conn.execute(
                    "SELECT 1 FROM steering_policies WHERE category = %s AND is_active = TRUE",
                    (category,),
                ).fetchone()
                should_activate = active_row is None

                pol_row = conn.execute(
                    """
                    INSERT INTO steering_policies (
                        name, category, version, model_id, layer_idx,
                        similarity_threshold, alpha, mode, method, is_active
                    )
                    VALUES (%s, %s, 1, %s, %s, %s, %s, 'prefill_only', 'additive', %s)
                    RETURNING id
                    """,
                    (
                        policy_name,
                        category,
                        artifact.model_id,
                        artifact.best_layer,
                        float(artifact.recommended_threshold),
                        float(default_alpha),
                        should_activate,
                    ),
                ).fetchone()
                assert pol_row is not None
                policy_id = pol_row["id"]

                steer_vec = np.asarray(
                    artifact.vector.detach().float().cpu().numpy(), dtype=np.float32
                )
                det_vec = np.asarray(
                    artifact.detector_vector.detach().float().cpu().numpy(), dtype=np.float32
                )
                conn.execute(
                    """
                    INSERT INTO steering_vectors (
                        policy_id, calibration_run_id, vector_dim, l2_norm,
                        eval_separation_score, embedding, detector_embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        policy_id,
                        run_id,
                        int(steer_vec.size),
                        float(artifact.raw_diff_norm),
                        float(best_score),
                        steer_vec,
                        det_vec,
                    ),
                )
                conn.commit()
                seeded.append(policy_name)

        return seeded

    def list_policies(self, active_only: bool = False) -> list[dict[str, Any]]:
        """Return policy metadata joined with vector and calibration lineage."""
        where = "WHERE p.is_active = TRUE" if active_only else ""
        sql = f"""
            SELECT
                p.id,
                p.name,
                p.category,
                p.version,
                p.model_id,
                p.layer_idx,
                p.similarity_threshold,
                p.alpha,
                p.mode,
                p.method,
                p.is_active,
                p.created_at,
                p.updated_at,
                sv.vector_dim,
                sv.l2_norm,
                sv.eval_separation_score,
                cr.id AS calibration_run_id,
                cr.pair_count
            FROM steering_policies p
            JOIN steering_vectors sv ON sv.policy_id = p.id
            JOIN calibration_runs cr ON cr.id = sv.calibration_run_id
            {where}
            ORDER BY p.category, p.name, p.version DESC
        """
        with self.connection() as conn:
            rows = conn.execute(sql).fetchall()
            conn.commit()
            return [dict(r) for r in rows]

    def get_active_policies_with_vectors(self) -> list[dict[str, Any]]:
        """Fetch active policies including serialized steering and detector vectors."""
        sql = """
            SELECT
                p.id,
                p.name,
                p.category,
                p.version,
                p.model_id,
                p.layer_idx,
                p.similarity_threshold,
                p.alpha,
                p.mode,
                p.method,
                p.is_active,
                p.updated_at,
                sv.l2_norm,
                sv.eval_separation_score,
                sv.embedding,
                sv.detector_embedding
            FROM steering_policies p
            JOIN steering_vectors sv ON sv.policy_id = p.id
            WHERE p.is_active = TRUE
            ORDER BY p.category, p.version DESC
        """
        with self.connection() as conn:
            rows = conn.execute(sql).fetchall()
            conn.commit()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["id"] = str(item["id"])
                item["updated_at"] = item["updated_at"].isoformat()
                item["embedding"] = _vector_to_list(item["embedding"])
                item["detector_embedding"] = _vector_to_list(item["detector_embedding"])
                result.append(item)
            return result

    def create_policy_version(
        self,
        name: str,
        category: str,
        similarity_threshold: float,
        alpha: float,
        layer_idx: int | None = None,
        mode: str = "prefill_only",
        method: str = "additive",
        activate: bool = True,
    ) -> dict[str, Any]:
        """Create a new versioned steering policy inheriting vectors from category's latest."""
        with self.connection() as conn:
            src = conn.execute(
                """
                SELECT
                    p.model_id,
                    p.layer_idx,
                    sv.calibration_run_id,
                    sv.vector_dim,
                    sv.l2_norm,
                    sv.eval_separation_score,
                    sv.embedding,
                    sv.detector_embedding
                FROM steering_policies p
                JOIN steering_vectors sv ON sv.policy_id = p.id
                WHERE p.category = %s
                ORDER BY p.is_active DESC, p.version DESC
                LIMIT 1
                """,
                (category,),
            ).fetchone()
            if src is None:
                raise ValueError(f"No existing calibrated vector found for category '{category}'")

            ver_row = conn.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next_ver
                FROM steering_policies WHERE name = %s
                """,
                (name,),
            ).fetchone()
            assert ver_row is not None
            next_version = int(ver_row["next_ver"])

            if activate:
                conn.execute(
                    """
                    UPDATE steering_policies
                    SET is_active = FALSE, updated_at = now()
                    WHERE category = %s AND is_active = TRUE
                    """,
                    (category,),
                )

            target_layer = layer_idx if layer_idx is not None else int(src["layer_idx"])
            pol = conn.execute(
                """
                INSERT INTO steering_policies (
                    name, category, version, model_id, layer_idx,
                    similarity_threshold, alpha, mode, method, is_active
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    name,
                    category,
                    next_version,
                    src["model_id"],
                    target_layer,
                    float(similarity_threshold),
                    float(alpha),
                    mode,
                    method,
                    activate,
                ),
            ).fetchone()
            assert pol is not None

            conn.execute(
                """
                INSERT INTO steering_vectors (
                    policy_id, calibration_run_id, vector_dim, l2_norm,
                    eval_separation_score, embedding, detector_embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    pol["id"],
                    src["calibration_run_id"],
                    src["vector_dim"],
                    src["l2_norm"],
                    src["eval_separation_score"],
                    src["embedding"],
                    src["detector_embedding"],
                ),
            )
            conn.commit()
            return dict(pol)

    def activate_policy(self, policy_id: UUID) -> dict[str, Any] | None:
        """Atomically activate a specific policy version within its category."""
        with self.connection() as conn:
            target = conn.execute(
                "SELECT id, category FROM steering_policies WHERE id = %s",
                (policy_id,),
            ).fetchone()
            if target is None:
                conn.commit()
                return None

            conn.execute(
                """
                UPDATE steering_policies
                SET is_active = FALSE, updated_at = now()
                WHERE category = %s AND is_active = TRUE
                """,
                (target["category"],),
            )
            updated = conn.execute(
                """
                UPDATE steering_policies
                SET is_active = TRUE, updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (policy_id,),
            ).fetchone()
            conn.commit()
            return dict(updated) if updated is not None else None

    def log_inference_audit(
        self,
        request_id: UUID,
        policy_id: UUID | None,
        model_id: str,
        prompt: str,
        prefill_cosine_sim: float,
        peak_decode_sim: float,
        intervened: bool,
        intervention_tokens: int,
        generated_tokens: int,
        ttft_ms: float,
        hook_avg_ms: float,
        hook_p99_ms: float,
        prefill_activation: Sequence[float] | None,
    ) -> None:
        """Persist request audit record and prefill activation snapshot into PostgreSQL."""
        digest, preview = _prompt_digest_and_preview(prompt)
        vec_arg = _to_vector_arg(prefill_activation)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO inference_audit_logs (
                    request_id, policy_id, model_id, prompt_sha256, prompt_preview,
                    prefill_cosine_sim, peak_decode_sim, intervened, intervention_tokens,
                    generated_tokens, ttft_ms, hook_avg_ms, hook_p99_ms, prefill_activation
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (request_id) DO NOTHING
                """,
                (
                    request_id,
                    policy_id,
                    model_id,
                    digest,
                    preview,
                    float(prefill_cosine_sim),
                    float(peak_decode_sim),
                    bool(intervened),
                    int(intervention_tokens),
                    int(generated_tokens),
                    float(ttft_ms),
                    float(hook_avg_ms),
                    float(hook_p99_ms),
                    vec_arg,
                ),
            )
            conn.commit()

    def get_drift_analytics(self, window_hours: int = 24, limit: int = 20) -> dict[str, Any]:
        """Compute policy telemetry rollups, rolling drift windows, and flagged events."""
        with self.connection() as conn:
            by_policy = conn.execute(
                """
                SELECT
                    COALESCE(p.name, 'baseline_unsteered') AS policy_name,
                    COALESCE(p.category, 'none') AS category,
                    COALESCE(p.version, 0) AS version,
                    COUNT(*) AS total_requests,
                    SUM(CASE WHEN l.intervened THEN 1 ELSE 0 END) AS intervened_requests,
                    ROUND(
                        (AVG(CASE WHEN l.intervened THEN 1.0 ELSE 0.0 END) * 100.0)::numeric, 2
                    )::float AS intervention_rate_pct,
                    ROUND(AVG(l.prefill_cosine_sim)::numeric, 4)::float AS avg_prefill_sim,
                    ROUND(AVG(l.ttft_ms)::numeric, 2)::float AS avg_ttft_ms,
                    ROUND(AVG(l.hook_avg_ms)::numeric, 4)::float AS avg_hook_ms,
                    ROUND(
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY l.hook_p99_ms)::numeric, 4
                    )::float AS p95_hook_p99_ms
                FROM inference_audit_logs l
                LEFT JOIN steering_policies p ON p.id = l.policy_id
                WHERE l.created_at >= now() - make_interval(hours => %s)
                GROUP BY p.name, p.category, p.version
                ORDER BY total_requests DESC
                """,
                (window_hours,),
            ).fetchall()

            hourly_drift = conn.execute(
                """
                WITH hourly AS (
                    SELECT
                        date_trunc('hour', created_at) AS bucket,
                        COUNT(*) AS requests,
                        AVG(prefill_cosine_sim) AS mean_prefill_sim,
                        AVG(CASE WHEN intervened THEN 1.0 ELSE 0.0 END) AS intervention_rate
                    FROM inference_audit_logs
                    WHERE created_at >= now() - make_interval(hours => %s)
                    GROUP BY 1
                )
                SELECT
                    bucket,
                    requests,
                    ROUND(mean_prefill_sim::numeric, 4)::float AS mean_prefill_sim,
                    ROUND(
                        AVG(mean_prefill_sim) OVER (
                            ORDER BY bucket ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                        )::numeric,
                        4
                    )::float AS rolling_6h_sim,
                    ROUND((intervention_rate * 100.0)::numeric, 2)::float AS intervention_rate_pct
                FROM hourly
                ORDER BY bucket DESC
                LIMIT 48
                """,
                (window_hours,),
            ).fetchall()

            recent_flagged = conn.execute(
                """
                SELECT
                    l.request_id,
                    COALESCE(p.name, 'baseline_unsteered') AS policy_name,
                    COALESCE(p.category, 'none') AS category,
                    l.prefill_cosine_sim,
                    l.peak_decode_sim,
                    l.intervention_tokens,
                    l.hook_avg_ms,
                    l.ttft_ms,
                    l.prompt_preview,
                    l.created_at
                FROM inference_audit_logs l
                LEFT JOIN steering_policies p ON p.id = l.policy_id
                WHERE l.intervened = TRUE
                ORDER BY l.created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            conn.commit()

        return {
            "window_hours": window_hours,
            "policy_breakdown": [dict(r) for r in by_policy],
            "hourly_drift": [dict(r) for r in hourly_drift],
            "recent_interventions": [dict(r) for r in recent_flagged],
        }

    def get_request_attribution(self, request_id: UUID, top_k: int = 5) -> dict[str, Any] | None:
        """Rank calibrated threat vectors and nearest historical requests via pgvector."""
        with self.connection() as conn:
            target = conn.execute(
                """
                SELECT
                    l.request_id,
                    l.model_id,
                    l.prompt_preview,
                    l.prefill_cosine_sim,
                    l.intervened,
                    l.prefill_activation,
                    l.created_at,
                    p.name AS policy_name
                FROM inference_audit_logs l
                LEFT JOIN steering_policies p ON p.id = l.policy_id
                WHERE l.request_id = %s
                """,
                (request_id,),
            ).fetchone()
            if target is None:
                conn.commit()
                return None

            if target["prefill_activation"] is None:
                conn.commit()
                return {
                    "request_id": str(target["request_id"]),
                    "policy_name": target["policy_name"],
                    "prompt_preview": target["prompt_preview"],
                    "intervened": target["intervened"],
                    "threat_matches": [],
                    "nearest_requests": [],
                }

            vec = target["prefill_activation"]
            threat_matches = conn.execute(
                """
                SELECT
                    p.id AS policy_id,
                    p.name AS policy_name,
                    p.category,
                    p.version,
                    ROUND(
                        (1.0 - (sv.detector_embedding <=> %s))::numeric, 4
                    )::float AS cosine_similarity
                FROM steering_vectors sv
                JOIN steering_policies p ON p.id = sv.policy_id
                ORDER BY sv.detector_embedding <=> %s
                LIMIT %s
                """,
                (vec, vec, top_k),
            ).fetchall()

            nearest_requests = conn.execute(
                """
                SELECT
                    l.request_id,
                    l.prompt_preview,
                    l.intervened,
                    l.prefill_cosine_sim,
                    ROUND(
                        (1.0 - (l.prefill_activation <=> %s))::numeric, 4
                    )::float AS activation_similarity,
                    l.created_at
                FROM inference_audit_logs l
                WHERE l.request_id <> %s
                  AND l.prefill_activation IS NOT NULL
                ORDER BY l.prefill_activation <=> %s
                LIMIT %s
                """,
                (vec, request_id, vec, top_k),
            ).fetchall()
            conn.commit()

        return {
            "request_id": str(target["request_id"]),
            "policy_name": target["policy_name"],
            "prompt_preview": target["prompt_preview"],
            "intervened": target["intervened"],
            "threat_matches": [dict(r) for r in threat_matches],
            "nearest_requests": [dict(r) for r in nearest_requests],
        }
