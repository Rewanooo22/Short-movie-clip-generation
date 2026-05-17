import math
from typing import Literal

import torch


def make_beta_schedule(timesteps: int, schedule: Literal["linear", "cosine"] = "linear") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class DiffusionProcess:
    def __init__(self, timesteps: int, schedule: str = "linear"):
        self.timesteps = timesteps
        betas = make_beta_schedule(timesteps, schedule)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].to(x0.device)[:, None, None, None, None]
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def get_training_target(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, prediction_target: str
    ) -> torch.Tensor:
        target = prediction_target.lower()
        if target == "eps":
            return noise
        if target == "x0":
            return x0
        if target == "v":
            a_bar = self.alpha_bars[t].to(x0.device)[:, None, None, None, None]
            return torch.sqrt(a_bar) * noise - torch.sqrt(1 - a_bar) * x0
        raise ValueError(f"Unsupported prediction_target: {prediction_target}")

    def _predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].to(xt.device)[:, None, None, None, None]
        return (xt - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar + 1e-8)

    def _predict_eps_from_x0(self, xt: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].to(xt.device)[:, None, None, None, None]
        return (xt - torch.sqrt(a_bar) * x0) / torch.sqrt(1 - a_bar + 1e-8)

    def _predict_eps_from_v(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].to(xt.device)[:, None, None, None, None]
        return torch.sqrt(a_bar) * v + torch.sqrt(1 - a_bar) * xt

    def cfg_predict(
        self,
        model,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond_text_emb: torch.Tensor,
        uncond_text_emb: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        pred_cond = model(xt, t, cond_text_emb)
        if guidance_scale <= 1.0:
            return pred_cond
        pred_uncond = model(xt, t, uncond_text_emb)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    def predict_eps(
        self,
        model,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond_text_emb: torch.Tensor,
        uncond_text_emb: torch.Tensor,
        guidance_scale: float,
        prediction_target: str,
    ) -> torch.Tensor:
        pred = self.cfg_predict(model, xt, t, cond_text_emb, uncond_text_emb, guidance_scale)
        target = prediction_target.lower()
        if target == "eps":
            return pred
        if target == "x0":
            return self._predict_eps_from_x0(xt, t, pred)
        if target == "v":
            return self._predict_eps_from_v(xt, t, pred)
        raise ValueError(f"Unsupported prediction_target: {prediction_target}")

    @torch.no_grad()
    def sample_ddpm_step(
        self,
        model,
        xt,
        t,
        cond_text_emb,
        uncond_text_emb,
        guidance_scale: float,
        prediction_target: str,
    ):
        eps = self.predict_eps(
            model, xt, t, cond_text_emb, uncond_text_emb, guidance_scale, prediction_target
        )
        a_t = self.alphas[t].to(xt.device)[:, None, None, None, None]
        a_bar_t = self.alpha_bars[t].to(xt.device)[:, None, None, None, None]
        beta_t = self.betas[t].to(xt.device)[:, None, None, None, None]
        x0 = self._predict_x0_from_eps(xt, t, eps).clamp(-1, 1)

        mean = (xt - ((1 - a_t) / torch.sqrt(1 - a_bar_t + 1e-8)) * eps) / torch.sqrt(a_t + 1e-8)
        noise = torch.randn_like(xt)
        nonzero = (t > 0).float()[:, None, None, None, None]
        sample = mean + nonzero * torch.sqrt(beta_t) * noise
        return sample, x0

    @torch.no_grad()
    def sample_ddim_step(
        self,
        model,
        xt,
        t,
        t_prev,
        cond_text_emb,
        uncond_text_emb,
        guidance_scale: float,
        prediction_target: str,
    ):
        eps = self.predict_eps(
            model, xt, t, cond_text_emb, uncond_text_emb, guidance_scale, prediction_target
        )
        x0 = self._predict_x0_from_eps(xt, t, eps).clamp(-1, 1)

        a_bar_t = self.alpha_bars[t].to(xt.device)[:, None, None, None, None]
        a_bar_prev = self.alpha_bars[t_prev].to(xt.device)[:, None, None, None, None]
        sample = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1 - a_bar_prev) * eps
        return sample, x0
