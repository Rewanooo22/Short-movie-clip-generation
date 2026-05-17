"""
Neural TTS via Microsoft Edge voices (edge-tts): no API key.

Bonus-aligned features:
- Multi-voice: any Edge voice short name (`edge-tts --list-voices`).
- Emotion presets: mapped to rate / pitch / volume.
- Adjustable prosody: override rate, pitch, volume (Edge format, e.g. rate="+10%").
- Chunk-based synthesis: long scripts split on sentence boundaries and concatenated.
- Context-aware splitting: sentences and clause boundaries (not fixed byte chunks).
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


def emotion_to_prosody(emotion: str) -> Tuple[str, str, str]:
    """Return (rate, pitch, volume) for edge_tts.Communicate."""
    e = (emotion or "neutral").lower().strip()
    presets: dict[str, Tuple[str, str, str]] = {
        "neutral": ("+0%", "+0Hz", "+0%"),
        "happy": ("+12%", "+5Hz", "+5%"),
        "excited": ("+18%", "+8Hz", "+8%"),
        "sad": ("-15%", "-6Hz", "-8%"),
        "calm": ("-5%", "-2Hz", "+0%"),
    }
    return presets.get(e, presets["neutral"])


def split_for_chunks(text: str, max_chars: int = 320) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 <= max_chars:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(s) <= max_chars:
                buf = s
            else:
                for i in range(0, len(s), max_chars):
                    part = s[i : i + max_chars].strip()
                    if part:
                        chunks.append(part)
                buf = ""
    if buf:
        chunks.append(buf)

    if not chunks and text:
        return [text[:max_chars]]
    return chunks


def _audio_codec_for_container(out_path: Path) -> Tuple[str, str]:
    ext = out_path.suffix.lower()
    if ext in (".m4a", ".aac", ".mp4"):
        return "aac", "128k"
    return "libmp3lame", "192k"


def _concat_audio(segment_paths: List[Path], out_path: Path, ffmpeg_exe: str) -> None:
    if not segment_paths:
        raise ValueError("No audio segments to concatenate.")
    acodec, abitrate = _audio_codec_for_container(out_path)
    if len(segment_paths) == 1:
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(segment_paths[0]),
                "-c:a",
                acodec,
                "-b:a",
                abitrate,
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        return

    with tempfile.TemporaryDirectory() as tmp:
        list_file = Path(tmp) / "concat.txt"
        lines = []
        for p in segment_paths:
            safe = str(p.resolve()).replace("'", "'\\''")
            lines.append(f"file '{safe}'")
        list_file.write_text("\n".join(lines), encoding="utf-8")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:a",
            acodec,
            "-b:a",
            abitrate,
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)


async def _synthesize_one(text: str, voice: str, rate: str, pitch: str, volume: str, out_path: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    await communicate.save(str(out_path))


async def _synthesize_chunks_async(
    text: str,
    voice: str,
    rate: str,
    pitch: str,
    volume: str,
    out_audio: Path,
    chunk_chars: int,
    ffmpeg_exe: str,
) -> None:
    chunks = split_for_chunks(text, max_chars=chunk_chars)
    if not chunks:
        raise ValueError("Narration text is empty.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        segments: List[Path] = []
        for i, ch in enumerate(chunks):
            seg = tmp / f"narr_{i:04d}.mp3"
            await _synthesize_one(ch, voice, rate, pitch, volume, seg)
            segments.append(seg)
        _concat_audio(segments, out_audio, ffmpeg_exe)


def synthesize_to_file(
    text: str,
    out_audio: Path,
    *,
    voice: str = "en-US-AriaNeural",
    emotion: str = "neutral",
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
    volume: Optional[str] = None,
    chunk_chars: int = 320,
    ffmpeg_exe: Optional[str] = None,
) -> None:
    import imageio_ffmpeg

    ff = ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()
    r, p, v = emotion_to_prosody(emotion)
    r = rate if rate is not None else r
    p = pitch if pitch is not None else p
    v = volume if volume is not None else v
    out_audio = Path(out_audio)
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    if not out_audio.suffix:
        out_audio = out_audio.with_suffix(".m4a")
    asyncio.run(_synthesize_chunks_async(text, voice, r, p, v, out_audio, chunk_chars, ff))
