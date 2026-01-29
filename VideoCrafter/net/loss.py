# loss.py
"""
NPNetV training losses.

Design goals:
- Robust supervision in latent space (Charbonnier loss).
- Explicit temporal smoothness regularization at low frequencies
  to suppress spatiotemporal flicker.
- Fully float32 computation for numerical stability.

This loss is designed for golden-pair supervision:
    x_T        : original noise
    x_T_target : golden / corrected noise
    x*_T       : NPNetV prediction
"""

import torch
import torch.nn as nn
import torch_dct


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (robust L1-like loss).

    L(x, y) = mean( sqrt((x - y)^2 + eps^2) )

    Compared to L2, this loss is less sensitive to outliers,
    which is important when supervising latent noise tensors.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class NPNetVLoss(nn.Module):
    """
    Composite loss for NPNetV.

    Components:
    1) Main reconstruction loss:
       - Charbonnier loss between predicted noise x*_T and target noise x_T_target.
    2) Temporal low-frequency regularization:
       - Enforces consistency of low-frequency temporal gradients
         to reduce flicker while preserving motion structure.

    Total loss:
        L = L_main + tau * L_temporal
    """

    def __init__(
        self,
        tau: float = 0.1,
        temporal_low_freq_k: int = 4,
        charbonnier_eps: float = 1e-6,
    ):
        """
        Args:
            tau: weight for temporal regularization term.
            temporal_low_freq_k: number of low-frequency DCT components
                                 kept along the temporal dimension.
            charbonnier_eps: epsilon for Charbonnier loss.
        """
        super().__init__()
        self.register_buffer("tau_default", torch.tensor(float(tau)))
        self.k = int(temporal_low_freq_k)
        self.main_loss = CharbonnierLoss(eps=charbonnier_eps)

    def _temporal_low_pass_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a temporal low-pass filter using DCT.

        Args:
            x: Tensor of shape (B, C, T, H, W)

        Returns:
            Low-frequency temporal reconstruction of x
            with the same shape (B, C, T, H, W).
        """
        # Move time dimension to the end for DCT
        x_perm = x.permute(0, 1, 3, 4, 2)  # (B, C, H, W, T)

        # DCT along temporal dimension
        x_freq = torch_dct.dct(x_perm, norm="ortho")

        # Keep only the lowest k temporal frequencies
        mask = torch.zeros_like(x_freq)
        mask[..., :self.k] = 1.0

        x_low = torch_dct.idct(x_freq * mask, norm="ortho")

        # Restore original layout
        x_low = x_low.permute(0, 1, 4, 2, 3).contiguous()
        return x_low

    def forward(
        self,
        x_star_T: torch.Tensor,
        x_target_T: torch.Tensor,
        tau_override: float | None = None,
    ):
        """
        Args:
            x_star_T: NPNetV predicted noise, shape (B, C, T, H, W)
            x_target_T: golden target noise, shape (B, C, T, H, W)
            tau_override: optional override for tau (e.g., curriculum scheduling)

        Returns:
            total_loss, main_loss, temporal_loss
        """
        # Ensure float32 for numerical stability
        x_star_T = x_star_T.float()
        x_target_T = x_target_T.float()

        # 1) Main robust reconstruction loss
        L_main = self.main_loss(x_star_T, x_target_T)

        # 2) Temporal low-frequency consistency loss
        x_star_low = self._temporal_low_pass_filter(x_star_T)
        x_target_low = self._temporal_low_pass_filter(x_target_T)

        # Temporal gradients
        d_pred = x_star_low[:, :, 1:] - x_star_low[:, :, :-1]
        d_gt = x_target_low[:, :, 1:] - x_target_low[:, :, :-1]

        L_temp = torch.mean((d_pred - d_gt) ** 2)

        # Combine losses
        tau = self.tau_default.item() if tau_override is None else float(tau_override)
        total = L_main + tau * L_temp

        return total, L_main, L_temp
