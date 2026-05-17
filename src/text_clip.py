from typing import List, Sequence

import torch


_clip_model = None
_clip_tokenizer = None


def _load_clip(device: torch.device):
    global _clip_model, _clip_tokenizer
    if _clip_model is not None:
        return _clip_model, _clip_tokenizer
    import open_clip

    _clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _clip_model.eval()
    return _clip_model, _clip_tokenizer


@torch.no_grad()
def encode_prompts_clip(prompts: Sequence[str], device) -> torch.Tensor:
    model, tokenizer = _load_clip(device)
    text_tokens = tokenizer(list(prompts)).to(device)
    text_features = model.encode_text(text_tokens).float()
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
    return text_features


def encode_prompts_for_model(prompts: List[str], device: torch.device, model_cfg: dict) -> torch.Tensor:
    mode = model_cfg.get("text_encoder", "hash")
    if mode == "clip":
        return encode_prompts_clip(prompts, device)
    from src.text import encode_prompts

    dim = int(model_cfg.get("text_emb_dim", model_cfg.get("time_emb_dim", 256)))
    return encode_prompts(prompts, dim, device)
