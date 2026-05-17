"""
Attach neural narration (Edge-TTS) to an existing silent video. See also: generate.py --narration.
"""

import argparse
import tempfile
from pathlib import Path

from src.audio import mux_video_audio, synthesize_to_file
from src.utils import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--text", type=str, default="", help="Narration string")
    p.add_argument("--text_file", type=str, default="", help="UTF-8 text file (overrides --text)")
    p.add_argument("--out", type=str, default="", help="Output path; default: <video_stem>_narrated.mp4")
    p.add_argument("--config", type=str, default="", help="Optional YAML to read audio.* defaults")
    p.add_argument("--voice", type=str, default="")
    p.add_argument("--emotion", type=str, default="neutral")
    p.add_argument("--rate", type=str, default="")
    p.add_argument("--pitch", type=str, default="")
    p.add_argument("--volume", type=str, default="")
    p.add_argument("--chunk_chars", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audio_cfg = {}
    if args.config:
        cfg = load_yaml(args.config)
        audio_cfg = cfg.get("audio", {}) or {}

    text = args.text.strip()
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()

    if not text:
        raise SystemExit("Provide --text or --text_file with non-empty content.")

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    voice = args.voice or audio_cfg.get("voice", "en-US-AriaNeural")
    emotion = args.emotion or audio_cfg.get("emotion", "neutral")
    chunk_chars = args.chunk_chars or int(audio_cfg.get("chunk_chars", 320))
    rate = args.rate or audio_cfg.get("rate") or None
    pitch = args.pitch or audio_cfg.get("pitch") or None
    volume = args.volume or audio_cfg.get("volume") or None

    out_path = Path(args.out) if args.out else video_path.with_name(f"{video_path.stem}_narrated.mp4")

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "narration.m4a"
        synthesize_to_file(
            text,
            audio_path,
            voice=voice,
            emotion=emotion,
            rate=rate,
            pitch=pitch,
            volume=volume,
            chunk_chars=chunk_chars,
        )
        mux_video_audio(video_path, audio_path, out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
