"""Model and tokenizer loading helpers for transformer residual-stream inspection."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(slots=True)
class LoadedModel:
    """Container pairing a loaded causal LM with its tokenizer and target device."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    num_layers: int


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve string dtype configuration to a torch.dtype."""
    if dtype_name not in DTYPE_MAP:
        supported = ", ".join(sorted(DTYPE_MAP))
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'. Expected one of: {supported}")
    return DTYPE_MAP[dtype_name]


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the sequential transformer block list from common causal LM architectures."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError(f"Cannot locate decoder layers on model type {type(model).__name__}")


def extract_hidden_tensor(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Extract the hidden-state tensor from either a Tensor or tuple layer output."""
    return output[0] if isinstance(output, tuple) else output


def replace_hidden_tensor(
    output: torch.Tensor | tuple[torch.Tensor, ...],
    new_hidden: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Return layer output with the hidden-state tensor replaced, preserving tuple structure."""
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    return new_hidden


def format_prompt(
    tokenizer: PreTrainedTokenizerBase,
    user_prompt: str,
    system_prompt: str = "You are a helpful, security-conscious engineering assistant.",
) -> str:
    """Format a user prompt using the model's chat template when available."""
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def load_model_and_tokenizer(
    model_id: str,
    dtype_name: str = "bfloat16",
    device_name: str = "cuda",
) -> LoadedModel:
    """Load model and tokenizer with low CPU RAM peak for local GPU execution."""
    device = torch.device(
        device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    dtype = (
        torch.float32
        if device.type == "cpu" and dtype_name == "bfloat16"
        else resolve_dtype(dtype_name)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    layers = get_decoder_layers(model)
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("Model config does not expose a positive hidden_size")

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        hidden_size=hidden_size,
        num_layers=len(layers),
    )
