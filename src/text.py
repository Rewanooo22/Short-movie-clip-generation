import hashlib
from typing import Sequence

import torch


def _token_to_index(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def encode_prompt(prompt: str, dim: int, device) -> torch.Tensor:
    vec = torch.zeros(dim, device=device, dtype=torch.float32)
    tokens = [t.strip().lower() for t in prompt.split() if t.strip()]
    if not tokens:
        return vec

    for token in tokens:
        idx = _token_to_index(token, dim)
        vec[idx] += 1.0
    return vec / (vec.norm(p=2) + 1e-6)


def encode_prompts(prompts: Sequence[str], dim: int, device) -> torch.Tensor:
    return torch.stack([encode_prompt(p, dim, device) for p in prompts], dim=0)
