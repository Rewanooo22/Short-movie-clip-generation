from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional


def _media_duration_seconds(path: Path, ffmpeg_exe: str) -> float:
    result = subprocess.run(
        [ffmpeg_exe, "-i", str(path)],
        capture_output=True,
        text=True,
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not m:
        raise RuntimeError(f"Could not read duration for {path} (is ffmpeg on PATH?)")
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


def mux_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    ffmpeg_exe: Optional[str] = None,
) -> None:
    """
    Mux TTS audio with video. Pads or trims audio to match video length.
    """
    import imageio_ffmpeg

    ff = ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()
    v_dur = _media_duration_seconds(video_path, ff)
    a_dur = _media_duration_seconds(audio_path, ff)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if a_dur <= 0:
        raise RuntimeError("Audio duration is zero; synthesis failed.")

    if abs(a_dur - v_dur) < 0.05:
        cmd = [
            ff,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return

    if a_dur < v_dur:
        pad = v_dur - a_dur + 0.1
        filt = f"[1:a]apad=pad_dur={pad}[aout]"
        cmd = [
            ff,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            filt,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-t",
            str(v_dur),
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return

    filt = f"[1:a]atrim=0:{v_dur},asetpts=PTS-STARTPTS[aout]"
    cmd = [
        ff,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-filter_complex",
        filt,
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-t",
        str(v_dur),
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
