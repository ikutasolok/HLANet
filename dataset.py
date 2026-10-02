"""Dataset helpers for prepared six-channel HLANet patches."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, manifest):
        path = Path(manifest)
        self.files = [Path(line.strip()) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.files = [file if file.is_absolute() else path.parent / file for file in self.files]
        if not self.files:
            raise ValueError(f'No patches in {path}')
        missing = [str(file) for file in self.files if not file.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as patch:
            image = np.asarray(patch['image'], dtype=np.float32)
            height = np.asarray(patch['height'], dtype=np.float32)
            valid = np.asarray(patch['valid'], dtype=bool) if 'valid' in patch else np.isfinite(height)
        if image.ndim != 3 or image.shape[0] != 6 or height.shape != image.shape[1:]:
            raise ValueError(f'Expected image [6,H,W] and height [H,W]: {self.files[index]}')
        valid &= np.isfinite(height) & np.isfinite(image).all(axis=0)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        height = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.zeros_like(height, dtype=np.int64)
        labels[(height > 0) & (height < 30)] = 1
        labels[height >= 30] = 2
        return {'image': torch.from_numpy(image), 'height': torch.from_numpy(height[None]),
                'valid': torch.from_numpy(valid[None]), 'class': torch.from_numpy(labels)}


class SyntheticDataset(Dataset):
    def __init__(self, high):
        self.high = high

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        generator = torch.Generator().manual_seed(index + (100 if self.high else 0))
        image = torch.randn((6, 64, 64), generator=generator)
        height = torch.randint(0, 25 if not self.high else 100, (64, 64), generator=generator).float()
        height[::3, ::3] = 0
        labels = torch.zeros_like(height, dtype=torch.long)
        labels[(height > 0) & (height < 30)] = 1
        labels[height >= 30] = 2
        return {'image': image, 'height': height[None], 'valid': torch.ones((1, 64, 64), dtype=torch.bool), 'class': labels}
