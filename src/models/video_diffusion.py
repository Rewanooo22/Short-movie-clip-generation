import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half_dim, device=t.device).float() / (half_dim - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x_in = x
        # Memory-safe temporal attention:
        # attend over time independently at each spatial location.
        x = self.norm(x).permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        head_dim = c // self.num_heads
        q = q.view(b * h * w, self.num_heads, head_dim, t)
        k = k.view(b * h * w, self.num_heads, head_dim, t)
        v = v.view(b * h * w, self.num_heads, head_dim, t)
        attn = torch.softmax(torch.einsum("bhct,bhcs->bhts", q, k) / math.sqrt(head_dim), dim=-1)
        out = torch.einsum("bhts,bhcs->bhct", attn, v).reshape(b * h * w, c, t)
        out = self.proj(out).reshape(b, h, w, c, t).permute(0, 3, 4, 1, 2)
        return out + x_in


@dataclass
class ModelConfig:
    in_channels: int = 3
    base_channels: int = 64
    time_emb_dim: int = 256
    text_emb_dim: int = 256


class VideoUNet3D(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ch = cfg.base_channels
        self.time_emb_dim = cfg.time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_emb_dim, cfg.time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(cfg.time_emb_dim * 2, cfg.time_emb_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.text_emb_dim, cfg.time_emb_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_emb_dim, cfg.time_emb_dim),
        )

        self.input_conv = nn.Conv3d(cfg.in_channels, ch, kernel_size=3, padding=1)

        self.down1 = ResBlock3D(ch, ch, cfg.time_emb_dim)
        self.down2 = ResBlock3D(ch, ch * 2, cfg.time_emb_dim)
        self.downsample = nn.Conv3d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1)

        self.mid = ResBlock3D(ch * 2, ch * 2, cfg.time_emb_dim)
        self.mid_attn = SelfAttention3D(ch * 2)

        self.upsample = nn.ConvTranspose3d(ch * 2, ch, kernel_size=4, stride=2, padding=1)
        self.up = ResBlock3D(ch * 2, ch, cfg.time_emb_dim)
        self.out = nn.Conv3d(ch, cfg.in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        if text_emb is not None:
            t_emb = t_emb + self.text_proj(text_emb)

        x0 = self.input_conv(x)
        d1 = self.down1(x0, t_emb)
        d2 = self.down2(d1, t_emb)
        ds = self.downsample(d2)

        m = self.mid_attn(self.mid(ds, t_emb))
        us = self.upsample(m)
        cat = torch.cat([us, d1], dim=1)
        u = self.up(cat, t_emb)
        return self.out(u)
