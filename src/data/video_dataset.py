from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.prompts import PROMPTS


class VideoPromptDataset(Dataset):
    """
    Expected dataset layout:
    - data/videos/*.mp4
    - data/videos/*.gif
    - data/videos/*/*.mp4
    """

    def __init__(self, root: str, channels: int, num_frames: int, frame_size: int):
        self.root = Path(root)
        self.channels = channels
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.video_paths = self._discover_videos()
        self.use_random = len(self.video_paths) == 0

    def _discover_videos(self) -> List[Path]:
        if not self.root.exists():
            return []
        exts = {".mp4", ".gif", ".avi", ".mov", ".mkv"}
        return [p for p in self.root.rglob("*") if p.suffix.lower() in exts]

    def __len__(self) -> int:
        return max(len(self.video_paths), 128)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame)
        img = img.resize((self.frame_size, self.frame_size), Image.BILINEAR)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[-1] > self.channels:
            arr = arr[..., : self.channels]
        if arr.shape[-1] < self.channels:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, self.channels - arr.shape[-1])), mode="edge")
        return arr

    def _sample_clip(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.random.randint(
                0, 255, size=(self.num_frames, self.frame_size, self.frame_size, self.channels), dtype=np.uint8
            )
        idxs = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
        sampled = [self._resize_frame(frames[i]) for i in idxs]
        return np.stack(sampled, axis=0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        if self.use_random:
            clip = torch.rand(self.channels, self.num_frames, self.frame_size, self.frame_size) * 2 - 1
            prompt = PROMPTS[index % len(PROMPTS)]
            return clip, prompt

        path = self.video_paths[index % len(self.video_paths)]
        frames = imageio.mimread(path)
        clip_np = self._sample_clip(frames).astype(np.float32) / 255.0
        clip_np = clip_np * 2.0 - 1.0
        clip = torch.from_numpy(clip_np).permute(3, 0, 1, 2).contiguous()

        prompt = path.stem.replace("_", " ")
        if not prompt.strip():
            prompt = PROMPTS[index % len(PROMPTS)]
        return clip, prompt