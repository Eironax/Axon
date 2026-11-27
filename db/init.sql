-- Axon Schema: Relational Policy Lineage, Audit Ledger, and pgvector Drift Analytics
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Calibration runs track which model, layer, and prompt pairs produced steering vectors
CREATE TABLE IF NOT EXISTS calibration_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id TEXT NOT NULL,
    layer_idx INTEGER NOT NULL,
    pair_count INTEGER NOT NULL CHECK (pair_count > 0),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS calibration_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES calibration_runs(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    positive_prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. Steering policies act as versioned firewall rules
CREATE TABLE IF NOT EXISTS steering_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    model_id TEXT NOT NULL,
    layer_idx INTEGER NOT NULL,
    similarity_threshold REAL NOT NULL CHECK (similarity_threshold >= -1.0 AND similarity_threshold <= 1.0),
    alpha REAL NOT NULL,
    mode TEXT NOT NULL DEFAULT 'prefill_only',
    method TEXT NOT NULL DEFAULT 'additive',
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);

-- 3. Dual vector storage (dim 1536 matches Qwen2.5-1.5B-Instruct hidden_size)
CREATE TABLE IF NOT EXISTS steering_vectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL REFERENCES steering_policies(id) ON DELETE CASCADE,
    calibration_run_id UUID NOT NULL REFERENCES calibration_runs(id) ON DELETE RESTRICT,
    vector_dim INTEGER NOT NULL DEFAULT 1536,
    l2_norm REAL NOT NULL,
    eval_separation_score REAL,
    embedding vector(1536) NOT NULL,
    detector_embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (policy_id)
);

-- 4. Immutable firewall audit ledger + async prefill activation snapshots for drift clustering
CREATE TABLE IF NOT EXISTS inference_audit_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    policy_id UUID REFERENCES steering_policies(id) ON DELETE SET NULL,
    model_id TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    prompt_preview TEXT NOT NULL,
    prefill_cosine_sim REAL,
    peak_decode_sim REAL,
    intervened BOOLEAN NOT NULL DEFAULT FALSE,
    intervention_tokens INTEGER NOT NULL DEFAULT 0,
    generated_tokens INTEGER NOT NULL DEFAULT 0,
    ttft_ms REAL,
    hook_avg_ms REAL,
    hook_p99_ms REAL,
    prefill_activation vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Forward-compatible additive column guards for existing volumes
ALTER TABLE steering_policies ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'prefill_only';
ALTER TABLE steering_policies ADD COLUMN IF NOT EXISTS method TEXT NOT NULL DEFAULT 'additive';
ALTER TABLE steering_vectors ADD COLUMN IF NOT EXISTS detector_embedding vector(1536);

-- Query & vector indexes
CREATE INDEX IF NOT EXISTS idx_policies_active
    ON steering_policies (is_active, category);

CREATE INDEX IF NOT EXISTS idx_audit_created_at
    ON inference_audit_logs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_intervened
    ON inference_audit_logs (intervened, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_steering_vectors_hnsw
    ON steering_vectors USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_steering_detector_hnsw
    ON steering_vectors USING hnsw (detector_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_audit_prefill_hnsw
    ON inference_audit_logs USING hnsw (prefill_activation vector_cosine_ops)
    WHERE prefill_activation IS NOT NULL;
