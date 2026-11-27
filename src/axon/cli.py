"""Command-line entrypoints for calibrating, evaluating, seeding, and serving Axon."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from axon.config import get_settings
from axon.engine.benchmark import calibrate_all_policies, run_eval_suite
from axon.engine.calibrate import load_calibration_artifact
from axon.engine.datasets import POLICY_DATASETS
from axon.engine.firewall import LatentFirewall, SteeringPolicy
from axon.engine.model import load_model_and_tokenizer
from axon.storage.cache import AxonRedisCache
from axon.storage.db import AxonDatabase


def _cmd_calibrate(args: argparse.Namespace) -> None:
    settings = get_settings()
    model_id = args.model_id or settings.model_id
    loaded = load_model_and_tokenizer(
        model_id, dtype_name=settings.torch_dtype, device_name=settings.device
    )
    categories = None if args.category == "all" else [args.category]
    results = calibrate_all_policies(
        loaded,
        model_id=model_id,
        output_dir=Path(args.output_dir),
        categories=categories,
    )

    for category, res in results.items():
        print(f"\n=== Calibration: {category} ({res.model_id}) ===")
        print(
            f"Best layer: {res.best_layer} | LOO threshold: {res.recommended_threshold:+.4f} "
            f"| CAA diff norm: {res.raw_diff_norm:.2f}"
        )
        print("Layer | Raw Sep | LOO Sep | LOO Acc | Threshold")
        for score in res.layer_scores:
            marker = " <-- best" if score.layer_idx == res.best_layer else ""
            print(
                f"{score.layer_idx:5d} | {score.raw_separation:7.4f} | {score.loo_separation:7.4f} "
                f"| {score.loo_accuracy:7.1%} | {score.recommended_threshold:+9.4f}{marker}"
            )


def _cmd_eval(args: argparse.Namespace) -> None:
    settings = get_settings()
    model_id = args.model_id or settings.model_id
    loaded = load_model_and_tokenizer(
        model_id, dtype_name=settings.torch_dtype, device_name=settings.device
    )
    summary = run_eval_suite(
        loaded=loaded,
        model_id=model_id,
        artifacts_dir=Path(args.artifacts_dir),
        alpha=args.alpha,
        mode=args.mode,
        method=args.method,
        max_new_tokens=args.max_new_tokens,
        sync_hook_timing=not args.no_sync_timing,
    )

    print(f"\n=== Axon Evaluation Summary ({summary.model_id} on {summary.device}) ===")
    print(f"Peak VRAM: {summary.peak_vram_mib:.1f} MiB")
    print(
        f"Mean prefill cos sim -> Risky: {summary.risky_mean_prefill_sim:+.4f} "
        f"| Benign: {summary.benign_mean_prefill_sim:+.4f}"
    )
    print(
        f"Risky intervention rate: {summary.risky_intervention_rate:.1%} "
        f"| Benign false-positive rate: {summary.benign_false_positive_rate:.1%}"
    )
    print(
        f"Hook latency -> Mean avg: {summary.mean_hook_avg_ms:.3f} ms "
        f"| Max p99: {summary.max_hook_p99_ms:.3f} ms"
    )
    print(
        f"Mean TTFT -> Baseline: {summary.mean_baseline_ttft_ms:.1f} ms "
        f"| Steered: {summary.mean_steered_ttft_ms:.1f} ms\n"
    )
    print("Case                       | Expected | PrefillSim | Thresh  | Intervened | HookAvg(ms)")
    for c in summary.cases:
        label = "RISKY" if c.expected_risky else "BENIGN"
        print(
            f"{c.name:26s} | {label:8s} | {c.prefill_cosine_sim:+10.4f} | "
            f"{c.threshold:+7.4f} | {str(c.intervened):10s} | {c.hook_avg_ms:10.3f}"
        )


def _cmd_generate(args: argparse.Namespace) -> None:
    settings = get_settings()
    model_id = args.model_id or settings.model_id
    loaded = load_model_and_tokenizer(
        model_id, dtype_name=settings.torch_dtype, device_name=settings.device
    )
    firewall = LatentFirewall(loaded.model, loaded.tokenizer, loaded.device, loaded.dtype)

    policy: SteeringPolicy | None = None
    if args.policy != "none":
        artifact = load_calibration_artifact(Path(args.artifacts_dir), args.policy)
        threshold = args.threshold if args.threshold is not None else artifact.recommended_threshold
        policy = SteeringPolicy(
            name=f"{args.policy}_v1",
            category=args.policy,
            layer_idx=artifact.best_layer,
            vector=artifact.vector,
            detector_vector=artifact.detector_vector,
            similarity_threshold=threshold,
            alpha=args.alpha,
            mode=args.mode,
            method=args.method,
        )

    result = firewall.generate(
        prompt=args.prompt,
        policy=policy,
        max_new_tokens=args.max_new_tokens,
        sync_hook_timing=True,
    )
    print(
        f"[policy={result.policy_name} prefill_sim={result.prefill_cosine_sim:+.4f} "
        f"intervened={result.intervened} hook_avg={result.hook_avg_ms:.3f}ms "
        f"ttft={result.ttft_ms:.1f}ms tps={result.tokens_per_sec:.1f}]\n"
    )
    print(result.response_text)


def _cmd_seed_db(args: argparse.Namespace) -> None:
    settings = get_settings()
    db = AxonDatabase(settings.postgres_dsn)
    cache = AxonRedisCache(settings.redis_url)
    try:
        db.ensure_schema()
        seeded = db.seed_from_artifacts(Path(args.artifacts_dir), POLICY_DATASETS)
        rows = db.get_active_policies_with_vectors()
        version = cache.publish_active_policies(rows)
        print(f"Seeded policies: {seeded or 'already present'} | Active Redis version: {version}")
    finally:
        cache.close()
        db.close()


def _cmd_serve(args: argparse.Namespace) -> None:
    uvicorn.run("axon.api.app:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    """Construct Axon CLI parser."""
    parser = argparse.ArgumentParser(prog="axon", description="Axon LLM Latent Firewall CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="Calibrate contrastive steering vectors")
    p_cal.add_argument(
        "--category", default="all", choices=["all", "unsafe_code", "prompt_injection"]
    )
    p_cal.add_argument("--output-dir", default="artifacts/vectors")
    p_cal.add_argument("--model-id", default=None)
    p_cal.set_defaults(func=_cmd_calibrate)

    p_eval = sub.add_parser("eval", help="Evaluate held-out prompts and latency")
    p_eval.add_argument("--artifacts-dir", default="artifacts/vectors")
    p_eval.add_argument("--alpha", type=float, default=20.0)
    p_eval.add_argument(
        "--mode",
        default="prefill_only",
        choices=["hybrid", "prefill_only", "decode_only", "always"],
    )
    p_eval.add_argument("--method", default="additive", choices=["additive", "projection"])
    p_eval.add_argument("--max-new-tokens", type=int, default=96)
    p_eval.add_argument("--no-sync-timing", action="store_true")
    p_eval.add_argument("--model-id", default=None)
    p_eval.set_defaults(func=_cmd_eval)

    p_gen = sub.add_parser("generate", help="Run single prompt through firewall")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument(
        "--policy", default="unsafe_code", choices=["unsafe_code", "prompt_injection", "none"]
    )
    p_gen.add_argument("--artifacts-dir", default="artifacts/vectors")
    p_gen.add_argument("--alpha", type=float, default=20.0)
    p_gen.add_argument("--threshold", type=float, default=None)
    p_gen.add_argument(
        "--mode",
        default="prefill_only",
        choices=["hybrid", "prefill_only", "decode_only", "always"],
    )
    p_gen.add_argument("--method", default="additive", choices=["additive", "projection"])
    p_gen.add_argument("--max-new-tokens", type=int, default=128)
    p_gen.add_argument("--model-id", default=None)
    p_gen.set_defaults(func=_cmd_generate)

    p_seed = sub.add_parser(
        "seed-db", help="Initialize schema and seed vectors into Postgres/Redis"
    )
    p_seed.add_argument("--artifacts-dir", default="artifacts/vectors")
    p_seed.set_defaults(func=_cmd_seed_db)

    p_srv = sub.add_parser("serve", help="Start FastAPI inference firewall server")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--reload", action="store_true")
    p_srv.set_defaults(func=_cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
