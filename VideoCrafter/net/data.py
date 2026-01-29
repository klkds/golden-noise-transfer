# data.py
"""
Load precomputed noise pairs from disk.

GitHub-friendly design:
- Dataset only loads tensors + prompt (no CLIP encoding here).
- Supports multiple key conventions to be compatible with different pair generators.
- safe_collate filters corrupted samples.

Expected per-.pt (recommended):
  - noise key:        x_T   (or z_T / xT)
  - target noise key: x_T_target (or z_T_target / xT_target / z_T_target)
  - prompt: str
  - meta: optional
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


# -----------------------------
# Key helpers
# -----------------------------
_NOISE_KEYS = ("x_T", "xT", "z_T")
_TARGET_KEYS = ("x_T_target", "xT_target", "z_T_target", "z_T_target")


def _pick_first(obj: Dict[str, Any], keys: Tuple[str, ...], name: str) -> torch.Tensor:
    for k in keys:
        if k in obj:
            v = obj[k]
            if not torch.is_tensor(v):
                raise TypeError(f"{name} field '{k}' is not a tensor: type={type(v)}")
            return v
    raise KeyError(f"Missing {name} key. Expected one of {keys}. Found keys={list(obj.keys())[:50]}")


def _normalize_noise_shape(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize noise to (C,T,H,W) on CPU float32.
    Accepts:
      - (C,T,H,W)
      - (B,C,T,H,W) with B==1
    """
    if x.dim() == 5:
        if x.shape[0] != 1:
            raise ValueError(f"Expected B==1 in saved pt, got shape {tuple(x.shape)}")
        x = x[0]
    if x.dim() != 4:
        raise ValueError(f"Expected noise tensor shape (C,T,H,W), got {tuple(x.shape)}")
    return x.float()


# -----------------------------
# Dataset
# -----------------------------
class PrecomputedNoiseDataset(Dataset):
    def __init__(self, directory: str):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        self.file_paths = sorted(self.directory.glob("*.pt"))
        if not self.file_paths:
            raise FileNotFoundError(
                f"No .pt files found in: {directory}. "
                f"Please check your config paths and file extensions."
            )

        print(f"[PrecomputedNoiseDataset] Found {len(self.file_paths)} samples in: {directory}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]

        try:
            obj: Dict[str, Any] = torch.load(file_path, map_location="cpu", weights_only=False)

            x = _pick_first(obj, _NOISE_KEYS, "noise")
            xt = _pick_first(obj, _TARGET_KEYS, "target")

            x = _normalize_noise_shape(x)
            xt = _normalize_noise_shape(xt)

            prompt = obj.get("prompt", "")
            if not isinstance(prompt, str):
                prompt = str(prompt)

            meta = obj.get("meta", {})

            # Return: (x_T, x_T_target, prompt, meta)
            return x, xt, prompt, meta

        except Exception as e:
            print(f"[PrecomputedNoiseDataset] Failed loading {file_path}: {e}")
            return None


def safe_collate(batch):
    """
    Filter out None samples.
    Returns None if batch becomes empty.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
