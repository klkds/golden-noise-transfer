# model.py
"""
NPNetV (Proposal Version)

Core ideas:
- Full 3D DCT over (T, H, W) with a soft low-pass mask in frequency space.
- Residual branch: 3D CNN with GroupNorm + GELU, FiLM conditioning, and SE.
- Temporal-frequency modulation: learns per-(channel, temporal-frequency) weights
  W_r(ft; E_txt) to modulate temporal DCT coefficients of the residual.
- Residual head is zero-initialized for stability.

Expected shapes:
- x_T:   (B, C, T, H, W)
- E_txt: (B, D_txt)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch_dct

import config


# -----------------------------
# Utilities
# -----------------------------
def _groups(c: int) -> int:
    if c >= 64:
        return 16
    if c >= 32:
        return 8
    return 4


def dct3(x: torch.Tensor) -> torch.Tensor:
    """
    3D DCT over (T, H, W) for x: (B, C, T, H, W).

    torch_dct only supports DCT on the last dimension, so we permute/transpose.
    """
    # DCT over W (last dim)
    x = torch_dct.dct(x, norm="ortho")

    # DCT over H
    x = x.transpose(-2, -1).contiguous()
    x = torch_dct.dct(x, norm="ortho")
    x = x.transpose(-2, -1).contiguous()

    # DCT over T
    x = x.transpose(-3, -1).contiguous()  # move T to last
    x = torch_dct.dct(x, norm="ortho")
    x = x.transpose(-3, -1).contiguous()  # move back

    return x


def idct3(x: torch.Tensor) -> torch.Tensor:
    """
    3D inverse DCT over (T, H, W) for x: (B, C, T, H, W).
    """
    # IDCT over W (last dim)
    x = torch_dct.idct(x, norm="ortho")

    # IDCT over H
    x = x.transpose(-2, -1).contiguous()
    x = torch_dct.idct(x, norm="ortho")
    x = x.transpose(-2, -1).contiguous()

    # IDCT over T
    x = x.transpose(-3, -1).contiguous()
    x = torch_dct.idct(x, norm="ortho")
    x = x.transpose(-3, -1).contiguous()

    return x


# -----------------------------
# Squeeze-Excite (3D)
# -----------------------------
class SqueezeExcite3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


# -----------------------------
# FiLM conditioning (text -> gamma, beta)
# -----------------------------
class FiLM(nn.Module):
    def __init__(self, text_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, T, H, W)
        cond: (B, D_txt)
        """
        B, C = x.shape[:2]
        gb = self.proj(cond)  # (B, 2C)
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.view(B, C, 1, 1, 1)
        beta = beta.view(B, C, 1, 1, 1)
        return (1.0 + gamma) * x + beta


# -----------------------------
# Residual 3D Block
# -----------------------------
class ResBlock3D(nn.Module):
    def __init__(self, channels: int, text_dim: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        p = dilation
        g = _groups(channels)

        self.conv1 = nn.Conv3d(channels, channels, 3, padding=p, dilation=dilation)
        self.norm1 = nn.GroupNorm(g, channels)

        self.conv2 = nn.Conv3d(channels, channels, 3, padding=p, dilation=dilation)
        self.norm2 = nn.GroupNorm(g, channels)

        self.se = SqueezeExcite3D(channels)
        self.film = FiLM(text_dim, channels)

        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, E_txt: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        h = self.se(h)
        h = self.film(h, E_txt)
        h = self.act(h)
        return x + h


# -----------------------------
# Residual Branch (CNN)
# -----------------------------
class ResidualBranch(nn.Module):
    def __init__(self, in_ch: int, text_dim: int, width: int = 64, depth: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )

        blocks = []
        for i in range(depth):
            dilation = 1 if i < (depth // 2) else 2
            blocks.append(ResBlock3D(width, text_dim, dilation=dilation, dropout=0.05))
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, in_ch, 1),
        )

        # Zero-init last layer for stable residual learning
        nn.init.zeros_(self.head[-1].weight)
        if self.head[-1].bias is not None:
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, E_txt: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h, E_txt)
        return self.head(h)


# -----------------------------
# Temporal-Frequency Modulation
# -----------------------------
class TemporalFrequencyModulator(nn.Module):
    def __init__(self, text_dim: int, channels: int, T: int):
        super().__init__()
        self.W_freq = nn.Linear(text_dim, channels * T)

    def forward(self, r: torch.Tensor, E_txt: torch.Tensor) -> torch.Tensor:
        """
        r: (B, C, T, H, W)

        Steps:
        - 1D DCT along temporal dimension T (per spatial location).
        - Predict per-(C, ft) weights from text embedding.
        - Multiply in temporal-frequency domain.
        - Inverse 1D DCT along T.
        """
        B, C, T, H, W = r.shape

        # DCT over T: move T to last dim for torch_dct
        r_perm = r.permute(0, 1, 3, 4, 2).contiguous()     # (B, C, H, W, T)
        r_freq_perm = torch_dct.dct(r_perm, norm="ortho")  # DCT over T
        r_freq = r_freq_perm.permute(0, 1, 4, 2, 3).contiguous()  # (B, C, T, H, W)

        # Text-conditioned temporal-frequency weights: (B, C, T, 1, 1)
        w = self.W_freq(E_txt).view(B, C, T, 1, 1)
        w = torch.sigmoid(w)

        mod = r_freq * w

        # IDCT over T: move T back to last dim
        mod_perm = mod.permute(0, 1, 3, 4, 2).contiguous()       # (B, C, H, W, T)
        r_out_perm = torch_dct.idct(mod_perm, norm="ortho")      # IDCT over T
        r_out = r_out_perm.permute(0, 1, 4, 2, 3).contiguous()   # (B, C, T, H, W)

        return r_out


# -----------------------------
# 3D Frequency Branch (soft low-pass)
# -----------------------------
class FrequencyBranch(nn.Module):
    def __init__(self, channels: int, decay: float = 1.0):
        super().__init__()
        self.channels = channels
        self.decay = float(decay)

    def _mask(self, T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        ft = torch.linspace(0.0, 1.0, T, device=device)
        fh = torch.linspace(0.0, 1.0, H, device=device)
        fw = torch.linspace(0.0, 1.0, W, device=device)

        Ft, Fh, Fw = torch.meshgrid(ft, fh, fw, indexing="ij")
        dist = Ft + Fh + Fw
        mask = torch.exp(-self.decay * dist)  # (T, H, W)

        # (1,1,T,H,W) for broadcasting over (B,C,*,*,*)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        spec = dct3(x)
        mask = self._mask(T, H, W, x.device)
        spec_low = spec * mask
        return idct3(spec_low)


# -----------------------------
# NPNetV (Proposal)
# -----------------------------
class NPNetV(nn.Module):
    def __init__(self, channels: int, T: int, H: int, W: int, freq_decay: float = 1.0):
        super().__init__()
        text_dim = int(config.TEXT_ENCODER_MODEL_DIM)

        self.freq_branch = FrequencyBranch(channels, decay=freq_decay)
        self.residual_branch = ResidualBranch(channels, text_dim, width=64, depth=8)
        self.tfreq_mod = TemporalFrequencyModulator(text_dim, channels, T)

        # Learnable mixing weights
        self.alpha = nn.Parameter(torch.tensor(0.6))  # frequency path weight
        self.beta = nn.Parameter(torch.tensor(0.8))   # residual path weight

    def forward(self, x_T: torch.Tensor, E_txt: torch.Tensor) -> torch.Tensor:
        """
        x_T:   (B, C, T, H, W)
        E_txt: (B, D_txt)
        """
        x_T = x_T.float()
        E_txt = E_txt.float()

        x_spec = self.freq_branch(x_T)                 # smooth low-pass component
        r_base = self.residual_branch(x_T, E_txt)      # learned residual
        r_mod = self.tfreq_mod(r_base, E_txt)          # temporal-frequency modulated residual

        # Output: a mixture of (low-pass) and (identity + modulated residual)
        return self.alpha * x_spec + self.beta * (x_T + r_mod)
