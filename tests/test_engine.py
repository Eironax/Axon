"""Fast CPU unit tests for calibration math and PyTorch forward-hook steering."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from axon.engine.calibrate import (
    calibrate_policy,
    compute_loo_separation,
    load_calibration_artifact,
    save_calibration_artifact,
)
from axon.engine.datasets import ContrastivePair
from axon.engine.firewall import LatentFirewall, SteeringPolicy


class _MockTokenizer:
    """Minimal deterministic tokenizer supporting batch encoding and generation decoding."""

    def __init__(self, risky_markers: tuple[str, ...] = ("RISKY",)) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.chat_template = None
        self.risky_markers = risky_markers

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if any(marker in text for marker in self.risky_markers):
            return [9, 9]
        return [3, 3]

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        ids = self.encode(text)
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(f"tok{t}" for t in token_ids if not (skip_special_tokens and t in (0, 1)))


class _DummyDecoderLayer(nn.Module):
    """Decoder block returning either a raw Tensor (transformers v5) or a tuple."""

    def __init__(
        self, hidden_dim: int, return_tuple: bool = False, bias_dir: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.return_tuple = return_tuple
        if bias_dir is not None:
            self.register_buffer("bias_dir", bias_dir)
        else:
            self.register_buffer("bias_dir", torch.zeros(hidden_dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, None]:
        out = hidden + self.bias_dir
        return (out, None) if self.return_tuple else out


class _DummyCausalLM(nn.Module):
    """Tiny mock causal LM compatible with LatentFirewall hooks and generate()."""

    def __init__(self, hidden_dim: int = 8, return_tuple: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_dim)
        self.embed = nn.Embedding(32, hidden_dim)
        # Token ids >= 7 map toward +e0 (risky), others toward -e0 (safe)
        with torch.no_grad():
            for idx in range(32):
                vec = torch.zeros(hidden_dim)
                vec[0] = 3.0 if idx >= 7 else -3.0
                vec[1] = 0.5
                self.embed.weight[idx] = vec

        layers = [_DummyDecoderLayer(hidden_dim, return_tuple=return_tuple) for _ in range(4)]
        self.model = SimpleNamespace(layers=nn.ModuleList(layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            out = layer(hidden)
            hidden = out[0] if isinstance(out, tuple) else out
        return SimpleNamespace(last_hidden_state=hidden)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 3,
        streamer=None,
        **_kwargs,
    ) -> torch.Tensor:
        # Prefill pass (seq_len > 1 or full prompt)
        self.forward(input_ids, attention_mask=attention_mask)
        generated = input_ids
        if streamer is not None:
            streamer.put(input_ids.cpu())
        for step in range(max_new_tokens):
            next_tok = torch.full((input_ids.shape[0], 1), 8 if step == 0 else 4, dtype=torch.long)
            self.forward(next_tok)
            generated = torch.cat([generated, next_tok], dim=-1)
            if streamer is not None:
                streamer.put(next_tok.cpu())
        if streamer is not None:
            streamer.end()
        return generated


def test_compute_loo_separation_perfect() -> None:
    pos = torch.tensor([[2.0, 0.2, 0.0], [2.5, -0.1, 0.0], [1.8, 0.0, 0.1]])
    neg = torch.tensor([[-2.0, 0.2, 0.0], [-2.5, -0.1, 0.0], [-1.8, 0.0, 0.1]])
    raw_sep, loo_sep, loo_acc, midpoint = compute_loo_separation(pos, neg)
    assert raw_sep > 1.8
    assert loo_sep > 1.8
    assert loo_acc == pytest.approx(1.0)
    assert abs(midpoint) < 0.1


def test_calibration_and_artifact_roundtrip(tmp_path: Path) -> None:
    model = _DummyCausalLM(hidden_dim=8)
    tokenizer = _MockTokenizer(risky_markers=("UNSAFE_CASE",))
    pairs = [
        ContrastivePair("unsafe_code", "UNSAFE_CASE one", "SAFE_CASE one"),
        ContrastivePair("unsafe_code", "UNSAFE_CASE two", "SAFE_CASE two"),
        ContrastivePair("unsafe_code", "UNSAFE_CASE three", "SAFE_CASE three"),
    ]
    res = calibrate_policy(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        category="unsafe_code",
        model_id="dummy-lm",
        device=torch.device("cpu"),
        candidate_layers=[1, 2],
    )
    assert res.best_layer in (1, 2)
    assert res.vector[0].item() > 0.9
    assert res.detector_vector[0].item() > 0.9

    save_calibration_artifact(res, tmp_path)
    loaded = load_calibration_artifact(tmp_path, "unsafe_code")
    assert loaded.best_layer == res.best_layer
    assert torch.allclose(loaded.vector, res.vector, atol=1e-6)
    assert torch.allclose(loaded.detector_vector, res.detector_vector, atol=1e-6)


@pytest.mark.parametrize("return_tuple", [False, True])
def test_firewall_hook_intervenes_and_reduces_threat_projection(return_tuple: bool) -> None:
    model = _DummyCausalLM(hidden_dim=8, return_tuple=return_tuple)
    tokenizer = _MockTokenizer()
    firewall = LatentFirewall(model, tokenizer, device=torch.device("cpu"), dtype=torch.float32)

    threat_vec = torch.zeros(8)
    threat_vec[0] = 1.0
    policy = SteeringPolicy(
        name="test_policy",
        category="unsafe_code",
        layer_idx=1,
        vector=threat_vec,
        similarity_threshold=0.5,
        alpha=1.0,
        mode="hybrid",
        method="projection",
    )

    captured_after: list[torch.Tensor] = []

    def inspect_next_layer(_mod, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured_after.append(h[:, -1, :].detach().clone())

    probe_handle = model.model.layers[2].register_forward_hook(inspect_next_layer)
    try:
        # Risky token id 8 triggers steering and should zero out positive +e0 projection
        risky_input = torch.tensor([[8, 8]], dtype=torch.long)
        with firewall.attach(policy) as session:
            model.forward(risky_input)
        trace = session.materialize()
        assert trace.intervened is True
        assert trace.prefill_cosine_sim > 0.9
        assert trace.prefill_activation is not None and len(trace.prefill_activation) == 8
        post_sim = F.cosine_similarity(captured_after[-1], threat_vec.unsqueeze(0), dim=-1).item()
        assert post_sim < 0.1

        # Safe token id 3 stays below threshold and must not intervene
        captured_after.clear()
        safe_input = torch.tensor([[3, 3]], dtype=torch.long)
        with firewall.attach(policy) as safe_session:
            model.forward(safe_input)
        safe_trace = safe_session.materialize()
        assert safe_trace.intervened is False
        assert safe_trace.prefill_cosine_sim < -0.5
    finally:
        probe_handle.remove()


def test_firewall_generate_end_to_end() -> None:
    model = _DummyCausalLM(hidden_dim=8, return_tuple=False)
    tokenizer = _MockTokenizer(risky_markers=("RISKY",))
    firewall = LatentFirewall(model, tokenizer, device=torch.device("cpu"), dtype=torch.float32)

    threat_vec = torch.zeros(8)
    threat_vec[0] = 1.0
    policy = SteeringPolicy(
        name="e2e_policy",
        category="unsafe_code",
        layer_idx=1,
        vector=threat_vec,
        similarity_threshold=0.5,
        alpha=1.0,
        mode="hybrid",
        method="projection",
    )
    result = firewall.generate("RISKY raw sql prompt", policy=policy, max_new_tokens=3)
    assert result.intervened is True
    assert result.intervention_tokens == 3
    assert result.generated_tokens > 0
    assert result.hook_avg_ms >= 0.0
    assert result.ttft_ms >= 0.0
