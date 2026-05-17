"""
Train phase-1 ablation configs, generate paired videos (fixed seed), run src.evaluate, write JSON summary.

Usage (from repo root):
  python scripts/collect_phase1_metrics.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "A rainy neon street in a futuristic city, camera slowly dollying forward"
REF_NAME = "_".join(PROMPT.lower().split()[:6]) + ".mp4"
REF_DIR = ROOT / "data" / "reference"
GEN_SEED = 42

RUNS = [
    ("baseline (hash, no smooth)", "configs/phase1_baseline.yaml", "checkpoints/phase1_baseline"),
    ("+ temporal smooth", "configs/phase1_smooth.yaml", "checkpoints/phase1_smooth"),
    ("+ CLIP text", "configs/phase1_clip.yaml", "checkpoints/phase1_clip"),
]


def ensure_reference_video() -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    path = REF_DIR / REF_NAME
    if path.exists():
        return path
    t, h, w = 8, 32, 32
    frames = []
    for ti in range(t):
        yy, xx = np.mgrid[0:h, 0:w]
        r = (np.sin(xx.astype(np.float32) * 0.2 + ti * 0.3) * 0.5 + 0.5) * 255.0
        g = (np.cos(yy.astype(np.float32) * 0.15 + ti * 0.25) * 0.5 + 0.5) * 255.0
        b = (np.sin((xx + yy).astype(np.float32) * 0.1 + ti * 0.2) * 0.5 + 0.5) * 255.0
        frame = np.stack([r, g, b], axis=-1).astype(np.uint8)
        frames.append(frame)
    imageio.mimwrite(str(path), frames, fps=8, quality=7)
    print(f"Wrote reference clip {path}")
    return path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    ensure_reference_video()
    device = "cuda"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    rows = []
    for label, cfg_rel, ckpt_dir_rel in RUNS:
        cfg = str(ROOT / cfg_rel)
        ckpt_dir = ROOT / ckpt_dir_rel
        cfg_dict = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
        epochs = int(cfg_dict["train"]["epochs"])
        run([sys.executable, "-m", "src.train", "--config", cfg])
        ckpt_path = ckpt_dir / f"epoch_{epochs}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint {ckpt_path}")
        run(
            [
                sys.executable,
                "-m",
                "src.generate",
                "--config",
                cfg,
                "--checkpoint",
                str(ckpt_path),
                "--prompt",
                PROMPT,
                "--seed",
                str(GEN_SEED),
            ]
        )
        pred_dir = ROOT / Path(cfg_dict["sample"]["output_dir"])
        out_json = ROOT / "outputs" / "phase1" / f"metrics_{Path(ckpt_dir_rel).name}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "-m",
                "src.evaluate",
                "--pred_dir",
                str(pred_dir),
                "--gt_dir",
                str(REF_DIR),
                "--text_prompt",
                PROMPT,
                "--device",
                device,
                "--output_json",
                str(out_json),
            ]
        )
        report = json.loads(out_json.read_text(encoding="utf-8"))
        rows.append(
            {
                "label": label,
                "metrics_json": str(out_json.relative_to(ROOT)),
                "LPIPS_temporal": report.get("LPIPS_temporal"),
                "FVD_proxy": report.get("FVD_proxy"),
                "CLIP_SIM": report.get("CLIP_SIM"),
            }
        )

    summary_path = ROOT / "outputs" / "phase1" / "table_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
