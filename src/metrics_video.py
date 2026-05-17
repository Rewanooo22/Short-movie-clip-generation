"""
Video / frame metrics for evaluation. Optional dependencies fail gracefully with None/skipped metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _to_nchw_float01(frames: np.ndarray) -> torch.Tensor:
    """frames: T,H,W,C uint8 -> float tensor T,3,H,W in [0,1]."""
    x = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    return x


def frechet_distance_np(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights

        # Some torchvision versions require aux_logits=True when loading IMAGENET1K_V1 weights.
        self.net = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
        self.net.fc = torch.nn.Identity()
        self.net.eval().to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        out = self.net(x)
        if isinstance(out, tuple):
            out = out[0]
        return out


@torch.no_grad()
def collect_inception_features(frames_nchw: torch.Tensor, device: torch.device, batch: int = 16) -> np.ndarray:
    model = InceptionFeatureExtractor(device)
    feats = []
    for i in range(0, frames_nchw.size(0), batch):
        chunk = frames_nchw[i : i + batch].to(device)
        feats.append(model(chunk).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fvd_proxy(pred: np.ndarray, gt: np.ndarray, device: torch.device) -> Optional[float]:
    try:
        import scipy  # noqa: F401
    except ImportError:
        return None
    t = min(pred.shape[0], gt.shape[0])
    pred = pred[:t]
    gt = gt[:t]
    if t < 2:
        return None
    pred_t = _to_nchw_float01(pred)
    gt_t = _to_nchw_float01(gt)
    f1 = collect_inception_features(pred_t, device)
    f2 = collect_inception_features(gt_t, device)
    mu1, sigma1 = f1.mean(axis=0), np.cov(f1, rowvar=False)
    mu2, sigma2 = f2.mean(axis=0), np.cov(f2, rowvar=False)
    return frechet_distance_np(mu1, sigma1, mu2, sigma2)


def compute_inception_score(frames_uint8: np.ndarray, device: torch.device) -> Optional[Tuple[float, float]]:
    try:
        from torchmetrics.image.inception import InceptionScore
    except ImportError:
        return None
    x = _to_nchw_float01(frames_uint8)
    if x.size(0) < 2:
        return None
    metric = InceptionScore(splits=10, normalize=True).to(device)
    metric.update(x.to(device))
    mean, std = metric.compute()
    return float(mean), float(std)


def compute_temporal_lpips(frames_uint8: np.ndarray, device: torch.device) -> Optional[float]:
    try:
        import lpips
    except ImportError:
        return None
    if frames_uint8.shape[0] < 2:
        return None
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()
    x = _to_nchw_float01(frames_uint8)
    x = x * 2.0 - 1.0
    scores = []
    with torch.no_grad():
        for i in range(x.size(0) - 1):
            a = x[i : i + 1].to(device)
            b = x[i + 1 : i + 2].to(device)
            scores.append(float(loss_fn(a, b).mean().item()))
    return float(np.mean(scores)) if scores else None


def compute_clip_similarity(frames_uint8: np.ndarray, text_prompt: str, device: torch.device) -> Optional[float]:
    try:
        import open_clip
        from PIL import Image
    except ImportError:
        return None
    if not text_prompt.strip():
        return None
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    text_tokens = tokenizer([text_prompt]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens).float()
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
        sims = []
        for i in range(frames_uint8.shape[0]):
            pil = Image.fromarray(frames_uint8[i])
            img_t = preprocess(pil).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_t).float()
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)
            sims.append(float((img_feat * text_feat).sum().item()))
    return float(np.mean(sims)) if sims else None


def _warp_with_flow(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """frame: 1,3,H,W; flow: 1,2,H,W in pixel units (RAFT convention)."""
    b, _, h, w = frame.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=frame.device),
        torch.linspace(-1, 1, w, device=frame.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    dx = 2 * flow[:, 0:1] / (w - 1)
    dy = 2 * flow[:, 1:2] / (h - 1)
    grid = grid + torch.cat([dx, dy], dim=-1).permute(0, 2, 3, 1)
    return F.grid_sample(frame, grid, align_corners=True, padding_mode="border")


@torch.no_grad()
def compute_flow_warping_error(frames_uint8: np.ndarray, device: torch.device) -> Optional[float]:
    if frames_uint8.shape[0] < 2:
        return None
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    except ImportError:
        return None
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights).to(device).eval()
    errs = []
    for i in range(frames_uint8.shape[0] - 1):
        f1 = torch.from_numpy(frames_uint8[i]).permute(2, 0, 1).float() / 255.0
        f2 = torch.from_numpy(frames_uint8[i + 1]).permute(2, 0, 1).float() / 255.0
        img1, img2 = transforms(f1, f2)
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)
        flows = model(img1, img2)
        flow = flows[-1]
        warped = _warp_with_flow(img2, flow)
        errs.append(float(F.l1_loss(warped, img1).item()))
    return float(np.mean(errs)) if errs else None


def compute_metrics_bundle(
    pred: np.ndarray,
    gt: np.ndarray,
    text_prompt: str,
    device: torch.device,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    fvd = compute_fvd_proxy(pred, gt, device)
    if fvd is not None:
        out["FVD_proxy"] = fvd
    iscore = compute_inception_score(pred, device)
    if iscore is not None:
        out["IS_mean"], out["IS_std"] = iscore
    lp = compute_temporal_lpips(pred, device)
    if lp is not None:
        out["LPIPS_temporal"] = lp
    if text_prompt.strip():
        cs = compute_clip_similarity(pred, text_prompt, device)
        if cs is not None:
            out["CLIP_SIM"] = cs
    fw = compute_flow_warping_error(pred, device)
    if fw is not None:
        out["FlowWarpingError"] = fw
    return out
