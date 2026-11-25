"""Contrastive activation calibration and layer-separation scoring."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from transformers import PreTrainedTokenizerBase

from axon.engine.datasets import ContrastivePair
from axon.engine.model import extract_hidden_tensor, format_prompt, get_decoder_layers


@dataclass(frozen=True, slots=True)
class LayerScore:
    """Leave-one-pair-out separation metrics for one transformer decoder layer."""

    layer_idx: int
    raw_separation: float
    loo_separation: float
    loo_accuracy: float
    recommended_threshold: float
    raw_diff_norm: float


@dataclass(slots=True)
class CalibrationResult:
    """Calibrated detector + CAA steering vectors and layer-sweep diagnostics."""

    category: str
    model_id: str
    best_layer: int
    vector: torch.Tensor
    detector_vector: torch.Tensor
    raw_diff_norm: float
    recommended_threshold: float
    layer_scores: list[LayerScore]


def compute_loo_separation(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Evaluate held-out separation via vectorized leave-one-pair-out validation.

    Returns ``(raw_sep, loo_sep, loo_acc, recommended_threshold)``, where
    ``recommended_threshold`` is chosen on LOO cosine scores to maximize balanced
    accuracy while prioritizing low false-positive rate on negative controls.
    """
    if pos_acts.shape != neg_acts.shape or pos_acts.ndim != 2:
        raise ValueError(
            f"Expected matching [N, D] tensors, got {pos_acts.shape} vs {neg_acts.shape}"
        )
    n_pairs = pos_acts.shape[0]
    if n_pairs < 2:
        raise ValueError("Need at least 2 contrastive pairs for leave-one-out evaluation")

    pos_f = pos_acts.float()
    neg_f = neg_acts.float()
    diffs = pos_f - neg_f
    total_diff = diffs.sum(dim=0)

    full_vec = F.normalize(total_diff / n_pairs, p=2, dim=0)
    raw_pos = F.cosine_similarity(pos_f, full_vec.unsqueeze(0), dim=-1)
    raw_neg = F.cosine_similarity(neg_f, full_vec.unsqueeze(0), dim=-1)
    raw_sep = float((raw_pos - raw_neg).mean().item())

    # Leave-one-pair-out: vector for sample i excludes diffs[i]
    loo_vecs = F.normalize((total_diff.unsqueeze(0) - diffs) / (n_pairs - 1), p=2, dim=-1)
    loo_pos = F.cosine_similarity(pos_f, loo_vecs, dim=-1)
    loo_neg = F.cosine_similarity(neg_f, loo_vecs, dim=-1)

    loo_sep = float((loo_pos - loo_neg).mean().item())

    # Choose threshold from LOO candidates favoring clean specificity over raw midpoint
    all_scores = torch.cat([loo_pos, loo_neg]).sort().values
    candidates = 0.5 * (all_scores[:-1] + all_scores[1:])
    best_thresh = float((0.5 * (loo_pos.mean() + loo_neg.mean())).item())
    best_score = -1.0

    for cand in candidates.tolist():
        tp = float((loo_pos >= cand).sum().item()) / n_pairs
        fp = float((loo_neg >= cand).sum().item()) / n_pairs
        # Weight specificity higher (1.2x) to protect benign traffic from false triggers
        objective = tp - 1.2 * fp
        if objective > best_score or (abs(objective - best_score) < 1e-6 and cand > best_thresh):
            best_score = objective
            best_thresh = float(cand)

    # Keep threshold slightly above negative tail when separation is cleanly positive
    neg_guard = float((loo_neg.mean() + 1.5 * loo_neg.std(unbiased=False)).item())
    threshold = max(best_thresh, min(neg_guard, float(loo_pos.mean().item())))
    correct = int((loo_pos >= threshold).sum().item() + (loo_neg < threshold).sum().item())
    loo_acc = correct / float(2 * n_pairs)

    return raw_sep, loo_sep, loo_acc, threshold


def default_candidate_layers(num_layers: int) -> list[int]:
    """Select middle-to-upper decoder layers where semantic features concentrate."""
    if num_layers <= 4:
        return list(range(num_layers))
    start = max(1, num_layers // 3)
    end = min(num_layers - 1, (3 * num_layers) // 4 + 1)
    return list(range(start, end))


@torch.inference_mode()
def collect_layer_activations(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    layer_indices: Sequence[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Run one forward pass per formatted text and capture last-token states across layers."""
    layers = get_decoder_layers(model)
    captured: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}
    current: dict[int, torch.Tensor] = {}

    def make_sniff_hook(layer_idx: int):
        def hook(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            hidden = extract_hidden_tensor(output)
            current[layer_idx] = hidden[:, -1, :].detach().float().cpu().squeeze(0)

        return hook

    handles = [layers[idx].register_forward_hook(make_sniff_hook(idx)) for idx in layer_indices]
    try:
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            current.clear()
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for idx in layer_indices:
                captured[idx].append(current[idx])
    finally:
        for handle in handles:
            handle.remove()

    return {idx: torch.stack(vecs, dim=0) for idx, vecs in captured.items()}


def _build_formatted_pair_texts(
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[ContrastivePair],
    system_prompt: str | None,
    use_responses: bool,
) -> tuple[list[str], list[str]]:
    pos_texts: list[str] = []
    neg_texts: list[str] = []
    for pair in pairs:
        pos_base = (
            format_prompt(tokenizer, pair.positive_prompt, system_prompt=system_prompt)
            if system_prompt is not None
            else format_prompt(tokenizer, pair.positive_prompt)
        )
        neg_base = (
            format_prompt(tokenizer, pair.negative_prompt, system_prompt=system_prompt)
            if system_prompt is not None
            else format_prompt(tokenizer, pair.negative_prompt)
        )
        if use_responses and pair.positive_response and pair.negative_response:
            # For CAA behavioral steering, hold prompt constant and vary only response prefix
            pos_texts.append(pos_base + pair.positive_response)
            neg_texts.append(pos_base + pair.negative_response)
        else:
            pos_texts.append(pos_base)
            neg_texts.append(neg_base)
    return pos_texts, neg_texts


def calibrate_policy(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[ContrastivePair],
    category: str,
    model_id: str,
    device: torch.device,
    candidate_layers: Sequence[int] | None = None,
    system_prompt: str | None = None,
) -> CalibrationResult:
    """Calibrate prompt-detector and CAA response-steering vectors across candidate layers."""
    if len(pairs) < 2:
        raise ValueError("Calibration requires at least 2 contrastive pairs")

    layers = get_decoder_layers(model)
    layer_indices = (
        list(candidate_layers)
        if candidate_layers is not None
        else default_candidate_layers(len(layers))
    )

    pos_prompts, neg_prompts = _build_formatted_pair_texts(
        tokenizer, pairs, system_prompt, use_responses=False
    )
    det_pos = collect_layer_activations(model, tokenizer, pos_prompts, layer_indices, device)
    det_neg = collect_layer_activations(model, tokenizer, neg_prompts, layer_indices, device)

    has_responses = all(p.positive_response and p.negative_response for p in pairs)
    if has_responses:
        steer_pos_txt, steer_neg_txt = _build_formatted_pair_texts(
            tokenizer, pairs, system_prompt, use_responses=True
        )
        steer_pos = collect_layer_activations(
            model, tokenizer, steer_pos_txt, layer_indices, device
        )
        steer_neg = collect_layer_activations(
            model, tokenizer, steer_neg_txt, layer_indices, device
        )
    else:
        steer_pos, steer_neg = det_pos, det_neg

    scores: list[LayerScore] = []
    detector_vectors: dict[int, torch.Tensor] = {}
    steering_vectors: dict[int, torch.Tensor] = {}

    for idx in layer_indices:
        raw_sep, loo_sep, loo_acc, threshold = compute_loo_separation(det_pos[idx], det_neg[idx])
        det_diff = (det_pos[idx] - det_neg[idx]).mean(dim=0)
        detector_vectors[idx] = F.normalize(det_diff, p=2, dim=0)

        steer_diff = (steer_pos[idx] - steer_neg[idx]).mean(dim=0)
        raw_norm = float(steer_diff.norm(p=2).item())
        steering_vectors[idx] = F.normalize(steer_diff, p=2, dim=0)

        scores.append(
            LayerScore(
                layer_idx=idx,
                raw_separation=raw_sep,
                loo_separation=loo_sep,
                loo_accuracy=loo_acc,
                recommended_threshold=threshold,
                raw_diff_norm=raw_norm,
            )
        )

    best = max(scores, key=lambda s: (s.loo_accuracy + 0.5 * max(0.0, s.loo_separation)))
    return CalibrationResult(
        category=category,
        model_id=model_id,
        best_layer=best.layer_idx,
        vector=steering_vectors[best.layer_idx],
        detector_vector=detector_vectors[best.layer_idx],
        raw_diff_norm=best.raw_diff_norm,
        recommended_threshold=best.recommended_threshold,
        layer_scores=scores,
    )


def save_calibration_artifact(result: CalibrationResult, output_dir: Path) -> tuple[Path, Path]:
    """Persist calibrated steering and detector vectors as safetensors + JSON metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"{result.category}.safetensors"
    meta_path = output_dir / f"{result.category}.json"

    save_safetensors(
        {
            "steering_vector": result.vector.detach().float().contiguous(),
            "detector_vector": result.detector_vector.detach().float().contiguous(),
        },
        str(tensor_path),
    )
    metadata = {
        "category": result.category,
        "model_id": result.model_id,
        "best_layer": result.best_layer,
        "vector_dim": int(result.vector.numel()),
        "raw_diff_norm": result.raw_diff_norm,
        "recommended_threshold": result.recommended_threshold,
        "layer_scores": [asdict(s) for s in result.layer_scores],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return tensor_path, meta_path


def load_calibration_artifact(output_dir: Path, category: str) -> CalibrationResult:
    """Load saved safetensors steering/detector vectors and JSON metadata sidecar."""
    tensor_path = output_dir / f"{category}.safetensors"
    meta_path = output_dir / f"{category}.json"
    tensors = load_safetensors(str(tensor_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    layer_scores = [LayerScore(**row) for row in metadata["layer_scores"]]
    steering = tensors["steering_vector"]
    detector = tensors.get("detector_vector", steering)
    return CalibrationResult(
        category=metadata["category"],
        model_id=metadata["model_id"],
        best_layer=int(metadata["best_layer"]),
        vector=steering,
        detector_vector=detector,
        raw_diff_norm=float(metadata["raw_diff_norm"]),
        recommended_threshold=float(metadata["recommended_threshold"]),
        layer_scores=layer_scores,
    )
