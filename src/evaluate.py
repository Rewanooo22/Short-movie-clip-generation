import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import imageio.v2 as imageio
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.metrics_video import compute_metrics_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="",
        help="Prompt for CLIP-SIM. If empty, uses prediction filename stem.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu (advanced metrics use this device).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="If set, write aggregate report JSON to this path.",
    )
    return parser.parse_args()


def load_video(path: Path) -> np.ndarray:
    frames = imageio.mimread(path)
    return np.stack(frames, axis=0)  # T,H,W,C


def paired_videos(pred_dir: Path, gt_dir: Path) -> List[tuple[Path, Path]]:
    pairs = []
    for pred in sorted(pred_dir.glob("*.mp4")):
        gt = gt_dir / pred.name
        if gt.exists():
            pairs.append((pred, gt))
    return pairs


def compute_basic_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    t = min(pred.shape[0], gt.shape[0])
    pred = pred[:t]
    gt = gt[:t]

    ssim_scores = []
    psnr_scores = []
    for i in range(t):
        pred_f = pred[i].astype(np.float32) / 255.0
        gt_f = gt[i].astype(np.float32) / 255.0
        ssim_scores.append(
            structural_similarity(pred_f, gt_f, data_range=1.0, channel_axis=-1)
        )
        psnr_scores.append(peak_signal_noise_ratio(gt_f, pred_f, data_range=1.0))

    temporal_lpips_proxy = float(np.mean(np.abs(np.diff(pred.astype(np.float32), axis=0)))) / 255.0
    return {
        "SSIM": float(np.mean(ssim_scores)),
        "PSNR": float(np.mean(psnr_scores)),
        "TemporalDiffProxy": temporal_lpips_proxy,
    }


def _pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    pairs = paired_videos(pred_dir, gt_dir)
    device = _pick_device(args.device)

    if not pairs:
        print("No matching predicted/ground-truth video pairs found.")
        return

    aggregate: Dict[str, List[float]] = {}
    per_video: List[Dict[str, Any]] = []

    for pred_path, gt_path in pairs:
        pred = load_video(pred_path)
        gt = load_video(gt_path)
        prompt = args.text_prompt.strip() or pred_path.stem.replace("_", " ")
        m: Dict[str, Any] = compute_basic_metrics(pred, gt)
        advanced = compute_metrics_bundle(pred, gt, prompt, device)
        m.update(advanced)
        per_video.append({"file": pred_path.name, "prompt": prompt, **m})
        for k, v in m.items():
            if k == "IS_std":
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                aggregate.setdefault(k, []).append(float(v))
        print(f"{pred_path.name}: {m}")

    report: Dict[str, Any] = {k: float(np.mean(v)) for k, v in aggregate.items()}
    report["per_video"] = per_video
    report["device"] = str(device)

    print("\nAggregate metrics:")
    for k, v in report.items():
        if k in ("per_video", "device"):
            continue
        print(f"- {k}: {v:.6f}")

    missing = []
    for key in ("FVD_proxy", "IS_mean", "CLIP_SIM", "LPIPS_temporal", "FlowWarpingError"):
        if key not in report:
            missing.append(key)
    if missing:
        print("\nOptional metrics skipped (install deps or add data):")
        for k in missing:
            print(f"- {k}")

    print("\nUser study: run a separate survey (≥10 raters); not automated here.")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to {out_path}")


if __name__ == "__main__":
    main()
