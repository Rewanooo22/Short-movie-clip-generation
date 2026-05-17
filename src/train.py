import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.diffusion import DiffusionProcess
from src.data import VideoPromptDataset
from src.models import ModelConfig, VideoUNet3D
from src.text_clip import encode_prompts_for_model
from src.utils import ensure_dir, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["project"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    diff_cfg = cfg["diffusion"]

    dataset = VideoPromptDataset(
        root=data_cfg["dataset_root"],
        channels=data_cfg["channels"],
        num_frames=data_cfg["num_frames"],
        frame_size=data_cfg["frame_size"],
    )
    if dataset.use_random:
        print(
            "[warning] No videos found in dataset_root. "
            "Training is using random synthetic clips, so generated videos will not be meaningful."
        )
    else:
        print(f"[info] Found {len(dataset.video_paths)} training videos in {data_cfg['dataset_root']}.")
    loader = DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=True)

    text_emb_dim = int(model_cfg.get("text_emb_dim", model_cfg["time_emb_dim"]))
    model = VideoUNet3D(
        ModelConfig(
            in_channels=data_cfg["channels"],
            base_channels=model_cfg["base_channels"],
            time_emb_dim=model_cfg["time_emb_dim"],
            text_emb_dim=text_emb_dim,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    criterion = nn.MSELoss()
    diffusion = DiffusionProcess(diff_cfg["timesteps"], diff_cfg["beta_schedule"])

    ckpt_dir = ensure_dir(train_cfg["checkpoint_dir"])
    cond_drop_prob = train_cfg.get("cond_drop_prob", 0.1)
    temporal_smooth_weight = float(train_cfg.get("temporal_smooth_weight", 0.0))
    prediction_target = diff_cfg.get("prediction_target", "eps")

    model.train()
    epoch_losses = []
    for epoch in range(train_cfg["epochs"]):
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}")
        running_loss = 0.0
        num_batches = 0
        for clips, prompts in loop:
            x0 = clips.to(device)
            b = x0.size(0)
            t = torch.randint(0, diff_cfg["timesteps"], (b,), device=device)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)
            prompt_list = list(prompts)
            drop_mask = torch.rand(b, device=device) < cond_drop_prob
            for i in range(b):
                if bool(drop_mask[i].item()):
                    prompt_list[i] = ""
            text_emb = encode_prompts_for_model(prompt_list, device, model_cfg)

            pred = model(xt, t, text_emb)
            target = diffusion.get_training_target(x0, noise, t, prediction_target)
            loss = criterion(pred, target)
            if temporal_smooth_weight > 0.0 and pred.shape[2] > 1:
                loss = loss + temporal_smooth_weight * ((pred[:, :, 1:] - pred[:, :, :-1]) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()

            loop.set_postfix(loss=float(loss.item()))
            running_loss += float(loss.item())
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        epoch_losses.append(epoch_loss)
        print(f"[epoch {epoch + 1}] mean_loss={epoch_loss:.6f}")

        if (epoch + 1) % train_cfg["save_every"] == 0:
            ckpt_path = Path(ckpt_dir) / f"epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "prediction_target": prediction_target,
                    "epoch_loss": epoch_loss,
                },
                ckpt_path,
            )

    summary_path = Path(ckpt_dir) / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prediction_target": prediction_target,
                "epochs": train_cfg["epochs"],
                "epoch_losses": epoch_losses,
                "final_loss": epoch_losses[-1] if epoch_losses else None,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
