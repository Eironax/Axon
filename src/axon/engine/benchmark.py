"""Calibration, held-out evaluation, and RTX 5060 latency benchmark helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from axon.engine.calibrate import (
    CalibrationResult,
    calibrate_policy,
    load_calibration_artifact,
    save_calibration_artifact,
)
from axon.engine.datasets import EVAL_CASES, POLICY_DATASETS, EvalPrompt
from axon.engine.firewall import LatentFirewall, SteeringMethod, SteeringMode, SteeringPolicy
from axon.engine.model import LoadedModel


@dataclass(slots=True)
class EvalCaseReport:
    """Comparison of unsteered baseline vs. Axon steered generation on one held-out prompt."""

    name: str
    category: str
    expected_risky: bool
    threshold: float
    prefill_cosine_sim: float
    peak_decode_sim: float
    intervened: bool
    intervention_tokens: int
    baseline_ttft_ms: float
    steered_ttft_ms: float
    baseline_tps: float
    steered_tps: float
    hook_avg_ms: float
    hook_p99_ms: float
    baseline_preview: str
    steered_preview: str


@dataclass(slots=True)
class BenchmarkSummary:
    """Aggregate accuracy, separation, latency, and VRAM metrics across held-out eval cases."""

    model_id: str
    device: str
    dtype: str
    peak_vram_mib: float
    risky_mean_prefill_sim: float
    benign_mean_prefill_sim: float
    risky_intervention_rate: float
    benign_false_positive_rate: float
    mean_hook_avg_ms: float
    max_hook_p99_ms: float
    mean_baseline_ttft_ms: float
    mean_steered_ttft_ms: float
    cases: list[EvalCaseReport]


def calibrate_all_policies(
    loaded: LoadedModel,
    model_id: str,
    output_dir: Path,
    categories: list[str] | None = None,
) -> dict[str, CalibrationResult]:
    """Calibrate and save steering vectors for selected policy categories."""
    selected = categories or list(POLICY_DATASETS.keys())
    results: dict[str, CalibrationResult] = {}

    for category in selected:
        if category not in POLICY_DATASETS:
            raise ValueError(f"Unknown calibration category '{category}'")
        pairs = POLICY_DATASETS[category]
        result = calibrate_policy(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            pairs=pairs,
            category=category,
            model_id=model_id,
            device=loaded.device,
        )
        save_calibration_artifact(result, output_dir)
        results[category] = result

    return results


def _preview(text: str, limit: int = 200) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def run_eval_suite(
    loaded: LoadedModel,
    model_id: str,
    artifacts_dir: Path,
    alpha: float = 20.0,
    mode: SteeringMode = "prefill_only",
    method: SteeringMethod = "additive",
    max_new_tokens: int = 96,
    sync_hook_timing: bool = True,
    cases: tuple[EvalPrompt, ...] = EVAL_CASES,
) -> BenchmarkSummary:
    """Run baseline vs. steered evaluation on held-out risky and benign control prompts."""
    firewall = LatentFirewall(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        device=loaded.device,
        dtype=loaded.dtype,
    )

    policies: dict[str, SteeringPolicy] = {}
    for category in {c.category for c in cases}:
        artifact = load_calibration_artifact(artifacts_dir, category)
        policies[category] = SteeringPolicy(
            name=f"{category}_v1",
            category=category,
            layer_idx=artifact.best_layer,
            vector=artifact.vector,
            detector_vector=artifact.detector_vector,
            similarity_threshold=artifact.recommended_threshold,
            alpha=alpha,
            mode=mode,
            method=method,
        )

    # Warm up CUDA kernels (baseline + hook path) before measuring latencies
    warmup_policy = next(iter(policies.values()))
    firewall.generate(prompt="Hello", policy=None, max_new_tokens=4, capture_prefill_vector=False)
    firewall.generate(
        prompt="Hello",
        policy=warmup_policy,
        max_new_tokens=4,
        capture_prefill_vector=False,
        sync_hook_timing=sync_hook_timing,
    )
    if loaded.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(loaded.device)

    reports: list[EvalCaseReport] = []
    for case in cases:
        policy = policies[case.category]
        baseline = firewall.generate(
            prompt=case.prompt,
            policy=None,
            max_new_tokens=max_new_tokens,
            capture_prefill_vector=False,
        )
        steered = firewall.generate(
            prompt=case.prompt,
            policy=policy,
            max_new_tokens=max_new_tokens,
            capture_prefill_vector=False,
            sync_hook_timing=sync_hook_timing,
        )
        reports.append(
            EvalCaseReport(
                name=case.name,
                category=case.category,
                expected_risky=case.expected_risky,
                threshold=policy.similarity_threshold,
                prefill_cosine_sim=steered.prefill_cosine_sim,
                peak_decode_sim=steered.peak_decode_sim,
                intervened=steered.intervened,
                intervention_tokens=steered.intervention_tokens,
                baseline_ttft_ms=baseline.ttft_ms,
                steered_ttft_ms=steered.ttft_ms,
                baseline_tps=baseline.tokens_per_sec,
                steered_tps=steered.tokens_per_sec,
                hook_avg_ms=steered.hook_avg_ms,
                hook_p99_ms=steered.hook_p99_ms,
                baseline_preview=_preview(baseline.response_text),
                steered_preview=_preview(steered.response_text),
            )
        )

    risky = [r for r in reports if r.expected_risky]
    benign = [r for r in reports if not r.expected_risky]
    peak_vram_mib = (
        torch.cuda.max_memory_allocated(loaded.device) / (1024.0 * 1024.0)
        if loaded.device.type == "cuda"
        else 0.0
    )

    risky_prefill = sum(r.prefill_cosine_sim for r in risky) / len(risky) if risky else 0.0
    benign_prefill = sum(r.prefill_cosine_sim for r in benign) / len(benign) if benign else 0.0
    risky_rate = (sum(1 for r in risky if r.intervened) / len(risky)) if risky else 0.0
    benign_fpr = (sum(1 for r in benign if r.intervened) / len(benign)) if benign else 0.0
    mean_hook = sum(r.hook_avg_ms for r in reports) / len(reports) if reports else 0.0
    max_p99 = max((r.hook_p99_ms for r in reports), default=0.0)
    mean_base_ttft = sum(r.baseline_ttft_ms for r in reports) / len(reports) if reports else 0.0
    mean_steer_ttft = sum(r.steered_ttft_ms for r in reports) / len(reports) if reports else 0.0

    summary = BenchmarkSummary(
        model_id=model_id,
        device=str(loaded.device),
        dtype=str(loaded.dtype),
        peak_vram_mib=peak_vram_mib,
        risky_mean_prefill_sim=risky_prefill,
        benign_mean_prefill_sim=benign_prefill,
        risky_intervention_rate=risky_rate,
        benign_false_positive_rate=benign_fpr,
        mean_hook_avg_ms=mean_hook,
        max_hook_p99_ms=max_p99,
        mean_baseline_ttft_ms=mean_base_ttft,
        mean_steered_ttft_ms=mean_steer_ttft,
        cases=reports,
    )

    report_path = artifacts_dir.parent / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary
