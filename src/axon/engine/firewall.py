"""Runtime PyTorch residual-stream firewall using forward hooks."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer

from axon.engine.model import (
    extract_hidden_tensor,
    format_prompt,
    get_decoder_layers,
    replace_hidden_tensor,
)

SteeringMode = Literal["hybrid", "prefill_only", "decode_only", "always"]
SteeringMethod = Literal["additive", "projection"]


@dataclass(frozen=True, slots=True)
class SteeringPolicy:
    """Policy parameters governing runtime activation monitoring and steering."""

    name: str
    category: str
    layer_idx: int
    vector: torch.Tensor
    detector_vector: torch.Tensor | None = None
    similarity_threshold: float = 0.05
    decode_threshold: float | None = None
    alpha: float = 20.0
    mode: SteeringMode = "prefill_only"
    method: SteeringMethod = "additive"

    def normalized_steering_vector(self) -> torch.Tensor:
        """Return 1D float32 unit-normalized steering vector."""
        vec = self.vector.detach().float().view(-1)
        return F.normalize(vec, p=2, dim=0)

    def normalized_detector_vector(self) -> torch.Tensor:
        """Return 1D float32 unit-normalized prefill detection vector."""
        base = self.detector_vector if self.detector_vector is not None else self.vector
        return F.normalize(base.detach().float().view(-1), p=2, dim=0)


@dataclass(slots=True)
class HookTrace:
    """Collected GPU activation telemetry materialized once after generation."""

    prefill_cosine_sim: float = 0.0
    peak_decode_sim: float = 0.0
    intervened: bool = False
    intervention_tokens: int = 0
    hook_avg_ms: float = 0.0
    hook_p99_ms: float = 0.0
    prefill_activation: list[float] | None = None


@dataclass(slots=True)
class FirewallResult:
    """Full generation output paired with latency and steering telemetry."""

    prompt: str
    response_text: str
    policy_name: str
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
    prefill_activation: list[float] | None = None


class _ActiveHookSession:
    """Per-generation hook state avoiding CPU-GPU sync inside token decoding."""

    def __init__(
        self,
        policy: SteeringPolicy,
        device: torch.device,
        dtype: torch.dtype,
        capture_prefill_vector: bool = True,
        sync_hook_timing: bool = False,
    ) -> None:
        self.policy = policy
        self.device = device
        self.dtype = dtype
        self.capture_prefill_vector = capture_prefill_vector
        self.sync_hook_timing = sync_hook_timing

        steer_unit = policy.normalized_steering_vector().to(device=device)
        det_unit = policy.normalized_detector_vector().to(device=device)
        self.steer_f32 = steer_unit.unsqueeze(0)  # [1, D]
        self.steer_layer = steer_unit.to(dtype=dtype).unsqueeze(0)  # [1, D]
        self.det_f32 = det_unit.unsqueeze(0)  # [1, D]

        self.decode_threshold = (
            policy.decode_threshold
            if policy.decode_threshold is not None
            else max(policy.similarity_threshold + 0.15, 0.18)
        )

        self.prefill_sim_tensor: torch.Tensor | None = None
        self.prefill_act_tensor: torch.Tensor | None = None
        self.prefill_triggered: torch.Tensor = torch.zeros(1, dtype=torch.bool, device=device)
        self.decode_sims: list[torch.Tensor] = []
        self.decode_interventions: list[torch.Tensor] = []
        self.hook_latencies_ms: list[float] = []

    def hook_fn(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        if self.sync_hook_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter_ns()

        hidden = extract_hidden_tensor(output)
        seq_len = hidden.shape[1]
        last_token = hidden[:, -1, :]  # [B, D]
        last_f32 = last_token.float()

        is_prefill = self.prefill_sim_tensor is None and (seq_len > 1 or not self.decode_sims)

        if is_prefill:
            cos_sim = F.cosine_similarity(last_f32, self.det_f32, dim=-1)
            self.prefill_sim_tensor = cos_sim.detach()
            if self.capture_prefill_vector:
                self.prefill_act_tensor = last_f32[0].detach().cpu()
            if self.policy.mode == "always":
                self.prefill_triggered = torch.ones_like(cos_sim, dtype=torch.bool)
            elif self.policy.mode in ("hybrid", "prefill_only"):
                self.prefill_triggered = cos_sim >= self.policy.similarity_threshold
            trigger_mask = self.prefill_triggered
        else:
            cos_sim = F.cosine_similarity(last_f32, self.steer_f32, dim=-1)
            self.decode_sims.append(cos_sim.detach())
            if self.policy.mode == "always":
                trigger_mask = torch.ones_like(cos_sim, dtype=torch.bool)
            elif self.policy.mode == "prefill_only":
                trigger_mask = self.prefill_triggered
            elif self.policy.mode == "decode_only":
                trigger_mask = cos_sim >= self.decode_threshold
            else:
                trigger_mask = self.prefill_triggered | (cos_sim >= self.decode_threshold)
            self.decode_interventions.append(trigger_mask.detach())

        # Compute steered candidate vector on-device without CPU-GPU sync
        if self.policy.method == "projection":
            proj = (last_f32 * self.steer_f32).sum(dim=-1, keepdim=True)
            positive_proj = torch.clamp(proj, min=0.0)
            delta_f32 = self.policy.alpha * positive_proj * self.steer_f32
            steered_last = (last_f32 - delta_f32).to(dtype=hidden.dtype)
        else:
            steered_last = last_token - (self.policy.alpha * self.steer_layer)

        updated_last = torch.where(trigger_mask.unsqueeze(-1), steered_last, last_token)
        steered_hidden = hidden.clone()
        steered_hidden[:, -1, :] = updated_last

        if self.sync_hook_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        self.hook_latencies_ms.append(elapsed_ms)

        return replace_hidden_tensor(output, steered_hidden)

    def materialize(self) -> HookTrace:
        """Pull accumulated scalar metrics to CPU once after generation completes."""
        prefill_sim = (
            float(self.prefill_sim_tensor.max().item())
            if self.prefill_sim_tensor is not None
            else 0.0
        )
        if self.decode_sims:
            stacked_sims = torch.stack(self.decode_sims)
            peak_decode = float(stacked_sims.max().item())
            intervention_tokens = int(torch.stack(self.decode_interventions).sum().item())
        else:
            peak_decode = prefill_sim
            intervention_tokens = int(self.prefill_triggered.sum().item())

        intervened = intervention_tokens > 0 or (
            self.prefill_triggered.any().item() if self.policy.mode != "decode_only" else False
        )

        latencies = sorted(self.hook_latencies_ms)
        if latencies:
            avg_ms = sum(latencies) / len(latencies)
            p99_idx = min(len(latencies) - 1, int(0.99 * (len(latencies) - 1)))
            p99_ms = latencies[p99_idx]
        else:
            avg_ms = 0.0
            p99_ms = 0.0

        prefill_vec = (
            self.prefill_act_tensor.tolist() if self.prefill_act_tensor is not None else None
        )
        return HookTrace(
            prefill_cosine_sim=prefill_sim,
            peak_decode_sim=peak_decode,
            intervened=bool(intervened),
            intervention_tokens=intervention_tokens,
            hook_avg_ms=avg_ms,
            hook_p99_ms=p99_ms,
            prefill_activation=prefill_vec,
        )


class LatentFirewall:
    """Thread-safe wrapper managing forward-hook attachment for causal LM generation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self._layers = get_decoder_layers(model)
        self._lock = threading.Lock()

    @contextmanager
    def attach(
        self,
        policy: SteeringPolicy,
        capture_prefill_vector: bool = True,
        sync_hook_timing: bool = False,
    ) -> Iterator[_ActiveHookSession]:
        """Attach a steering hook to the policy's target layer for the duration of a call."""
        if not (0 <= policy.layer_idx < len(self._layers)):
            num_layers = len(self._layers)
            raise IndexError(
                f"layer_idx={policy.layer_idx} out of range for model with {num_layers} layers"
            )
        session = _ActiveHookSession(
            policy=policy,
            device=self.device,
            dtype=self.dtype,
            capture_prefill_vector=capture_prefill_vector,
            sync_hook_timing=sync_hook_timing,
        )
        handle = self._layers[policy.layer_idx].register_forward_hook(session.hook_fn)
        try:
            yield session
        finally:
            handle.remove()

    def generate(
        self,
        prompt: str,
        policy: SteeringPolicy | None = None,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        capture_prefill_vector: bool = True,
        sync_hook_timing: bool = False,
    ) -> FirewallResult:
        """Run steered (or baseline when policy is None) generation and collect telemetry."""
        formatted = (
            format_prompt(self.tokenizer, prompt, system_prompt=system_prompt)
            if system_prompt is not None
            else format_prompt(self.tokenizer, prompt)
        )
        encoded = self.tokenizer(formatted, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
        else:
            # Clear sampling defaults from model generation_config when greedy decoding
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        with self._lock:
            start_ns = time.perf_counter_ns()
            first_token_ns: int | None = None
            chunks: list[str] = []
            gen_error: list[BaseException] = []

            def run_generation(active_policy: SteeringPolicy | None) -> HookTrace | None:
                if active_policy is None:
                    self.model.generate(**gen_kwargs)
                    return None
                with self.attach(
                    active_policy,
                    capture_prefill_vector=capture_prefill_vector,
                    sync_hook_timing=sync_hook_timing,
                ) as session:
                    self.model.generate(**gen_kwargs)
                return session.materialize()

            trace_holder: list[HookTrace | None] = [None]

            def worker() -> None:
                try:
                    with torch.inference_mode():
                        trace_holder[0] = run_generation(policy)
                except BaseException as exc:  # pragma: no cover - surfaced on main thread
                    gen_error.append(exc)
                    streamer.end()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            for text_chunk in streamer:
                if first_token_ns is None and text_chunk:
                    first_token_ns = time.perf_counter_ns()
                chunks.append(text_chunk)
            thread.join()

            if gen_error:
                raise gen_error[0]

            end_ns = time.perf_counter_ns()

        response_text = "".join(chunks).strip()
        generated_tokens = len(self.tokenizer.encode(response_text, add_special_tokens=False))
        total_ms = (end_ns - start_ns) / 1_000_000.0
        ttft_ms = (
            (first_token_ns - start_ns) / 1_000_000.0 if first_token_ns is not None else total_ms
        )
        tps = (generated_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        trace = trace_holder[0] or HookTrace()
        return FirewallResult(
            prompt=prompt,
            response_text=response_text,
            policy_name=policy.name if policy is not None else "baseline_unsteered",
            prefill_cosine_sim=trace.prefill_cosine_sim,
            peak_decode_sim=trace.peak_decode_sim,
            intervened=trace.intervened,
            intervention_tokens=trace.intervention_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=ttft_ms,
            total_generation_ms=total_ms,
            tokens_per_sec=tps,
            hook_avg_ms=trace.hook_avg_ms,
            hook_p99_ms=trace.hook_p99_ms,
            prefill_activation=trace.prefill_activation,
        )
