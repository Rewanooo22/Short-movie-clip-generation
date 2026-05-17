import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.utils import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["eps", "x0", "v"],
        help="Prediction targets to compare.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ablation",
        help="Where to store generated configs and summary files.",
    )
    return parser.parse_args()


def latest_checkpoint_epoch(checkpoint_dir: Path) -> Optional[int]:
    epochs = []
    for ckpt in checkpoint_dir.glob("epoch_*.pt"):
        stem = ckpt.stem  # epoch_X
        try:
            epochs.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(epochs) if epochs else None


def run_train(config_path: Path) -> None:
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)

    out_dir = ensure_dir(args.output_dir)
    cfg_dir = ensure_dir(str(Path(out_dir) / "configs"))
    rows = []

    for target in args.targets:
        run_cfg = copy.deepcopy(base_cfg)
        run_cfg["diffusion"]["prediction_target"] = target
        run_cfg["train"]["checkpoint_dir"] = str(Path("checkpoints") / f"ablation_{target}")

        cfg_path = Path(cfg_dir) / f"ablation_{target}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump(run_cfg, f, sort_keys=False)

        print(f"\n=== Running ablation for target={target} ===")
        run_train(cfg_path)

        ckpt_dir = Path(run_cfg["train"]["checkpoint_dir"])
        summary_path = ckpt_dir / "train_summary.json"

        final_loss = None
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            final_loss = summary.get("final_loss")

        rows.append(
            {
                "prediction_target": target,
                "checkpoint_dir": str(ckpt_dir),
                "last_epoch": latest_checkpoint_epoch(ckpt_dir),
                "final_train_loss": final_loss,
            }
        )

    json_path = Path(out_dir) / "target_ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = Path(out_dir) / "target_ablation_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prediction_target", "checkpoint_dir", "last_epoch", "final_train_loss"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nAblation completed.")
    print(f"- JSON: {json_path}")
    print(f"- CSV:  {csv_path}")


if __name__ == "__main__":
    main()
