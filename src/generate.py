import argparse
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from src.diffusion import DiffusionProcess
from src.data.prompts import THEME_NAME
from src.models import ModelConfig, VideoUNet3D
from src.text_clip import encode_prompts_for_model
from src.utils import ensure_dir, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="If >= 0, fix RNGs before sampling initial noise (reproducible comparisons).",
    )
    parser.add_argument(
        "--narration",
        type=str,
        default="",
        help="Optional narration script; synthesized with Edge-TTS and muxed (bonus module).",
    )
    parser.add_argument("--voice", type=str, default="", help="Edge voice short name, e.g. en-US-GuyNeural")
    parser.add_argument("--emotion", type=str, default="", help="neutral|happy|excited|sad|calm")
    parser.add_argument("--skip_audio", action="store_true", help="Disable audio even if config enables it")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.seed >= 0:
        set_seed(args.seed)
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    sample_cfg = cfg["sample"]
    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]
    audio_cfg = cfg.get("audio", {}) or {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_emb_dim = int(model_cfg.get("text_emb_dim", model_cfg["time_emb_dim"]))
    model = VideoUNet3D(
        ModelConfig(
            in_channels=data_cfg["channels"],
            base_channels=model_cfg["base_channels"],
            time_emb_dim=model_cfg["time_emb_dim"],
            text_emb_dim=text_emb_dim,
        )
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = DiffusionProcess(diff_cfg["timesteps"], diff_cfg["beta_schedule"])
    guidance_scale = float(sample_cfg.get("guidance_scale", 1.0))
    prediction_target = diff_cfg.get("prediction_target", "eps")

    cond_text_emb = encode_prompts_for_model([args.prompt], device, model_cfg)
    uncond_text_emb = encode_prompts_for_model([""], device, model_cfg)

    # Baseline random latent initialization for demonstration.
    x = torch.randn(
        1,
        data_cfg["channels"],
        data_cfg["num_frames"],
        data_cfg["frame_size"],
        data_cfg["frame_size"],
        device=device,
    )

    steps = sample_cfg["num_steps"]
    ts = torch.linspace(diff_cfg["timesteps"] - 1, 0, steps).long().to(device)
    sampler = sample_cfg.get("sampler", "ddim").lower()
    for i, t in enumerate(ts):
        t_batch = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
        if sampler == "ddpm":
            x, _ = diffusion.sample_ddpm_step(
                model,
                x,
                t_batch,
                cond_text_emb,
                uncond_text_emb,
                guidance_scale,
                prediction_target,
            )
        else:
            t_prev_value = int(ts[i + 1].item()) if i + 1 < len(ts) else 0
            t_prev = torch.full((1,), t_prev_value, device=device, dtype=torch.long)
            x, _ = diffusion.sample_ddim_step(
                model,
                x,
                t_batch,
                t_prev,
                cond_text_emb,
                uncond_text_emb,
                guidance_scale,
                prediction_target,
            )

    out_dir = ensure_dir(sample_cfg["output_dir"])
    name = "_".join(args.prompt.lower().split()[:6]) or "sample"
    out_path = Path(out_dir) / f"{name}.mp4"

    clip = x.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # T,H,W,C
    clip = np.clip((clip + 1) / 2, 0, 1)
    clip = (clip * 255).astype(np.uint8)

    imageio.mimwrite(str(out_path), clip, fps=8, quality=7)
    print(f"Theme: {THEME_NAME}")
    print(f"Prompt: {args.prompt}")
    print(f"Saved video to: {out_path}")

    narration = (args.narration or "").strip()
    if not narration and audio_cfg.get("narration_from_prompt"):
        narration = args.prompt.strip()
    want_audio = (not args.skip_audio) and (bool(narration) or bool(audio_cfg.get("enable")))
    if want_audio and narration:
        try:
            from src.audio import mux_video_audio, synthesize_to_file
        except ImportError as e:
            print(f"Audio skipped (import error): {e}")
            return
        voice = args.voice or audio_cfg.get("voice", "en-US-AriaNeural")
        emotion = (args.emotion or audio_cfg.get("emotion") or "neutral").strip()
        chunk_chars = int(audio_cfg.get("chunk_chars", 320))
        rate = audio_cfg.get("rate") or None
        pitch = audio_cfg.get("pitch") or None
        volume = audio_cfg.get("volume") or None
        out_narrated = out_path.with_name(f"{out_path.stem}_narrated{out_path.suffix}")
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "narration.m4a"
            synthesize_to_file(
                narration,
                audio_path,
                voice=voice,
                emotion=emotion,
                rate=rate,
                pitch=pitch,
                volume=volume,
                chunk_chars=chunk_chars,
            )
            mux_video_audio(out_path, audio_path, out_narrated)
        print(f"Saved video with narration: {out_narrated}")
    elif want_audio and not narration:
        print("Audio enabled in config but no narration text; set --narration or audio.narration_from_prompt.")


if __name__ == "__main__":
    main()
