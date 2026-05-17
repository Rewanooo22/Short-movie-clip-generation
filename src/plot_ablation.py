import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/ablation/target_ablation_summary.csv",
    )
    parser.add_argument("--out", type=str, default="outputs/ablation/target_ablation_loss.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.csv)
    if not path.exists():
        print(f"CSV not found: {path}")
        return
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    labels = [r["prediction_target"] for r in rows]
    losses = []
    for r in rows:
        v = r.get("final_train_loss")
        losses.append(float(v) if v not in (None, "", "nan") else 0.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(labels, losses, color=["#4c72b0", "#55a868", "#c44e52"])
    plt.ylabel("Final train loss (mean MSE)")
    plt.xlabel("Prediction target")
    plt.title("Objective ablation (lower is better for this proxy)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
