"""
Generate golden noise pairs using VideoCrafter + DDIM forward/inversion.

This script:
- Samples random x_T
- Runs DDIM forward K steps (denoising, strong CFG)
- Runs DDIM inversion K steps (re-noising, weak CFG)
- Saves (x_T, x_T_target, prompt, meta) as supervision for NPNet training

Key features:
- Uses global DDIM indices (corrected version)
- Supports resume from existing .pt files
- No hard-coded paths
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List

import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

# -----------------------------
# Utilities
# -----------------------------
def add_repo_to_syspath(repo_root: str) -> None:
    repo_root = os.path.abspath(repo_root)
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"Repo root not found: {repo_root}")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


# -----------------------------
# DDIM forward / inversion
# -----------------------------
@torch.no_grad()
def ddim_forward_K_steps(
    sampler,
    x_T: torch.Tensor,
    cond,
    uc,
    cfg_scale: float,
    K_steps: int,
):
    """
    DDIM forward denoising for K steps using global DDIM indices.
    """
    device = x_T.device
    B = x_T.shape[0]

    total_steps = len(sampler.ddim_timesteps)
    if K_steps > total_steps:
        raise ValueError(f"K_steps={K_steps} > total_ddim_steps={total_steps}")

    start_idx = total_steps - K_steps
    x_t = x_T

    for global_idx in range(start_idx, total_steps):
        t = sampler.ddim_timesteps[global_idx]
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        alpha_t = sampler.ddim_alphas[global_idx]
        alpha_prev = sampler.ddim_alphas_prev[global_idx]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = sampler.ddim_sqrt_one_minus_alphas[global_idx]

        if cfg_scale > 1.0 and uc is not None:
            eps_c = sampler.model.apply_model(x_t, t_tensor, cond)
            eps_u = sampler.model.apply_model(x_t, t_tensor, uc)
            eps = eps_u + cfg_scale * (eps_c - eps_u)
        else:
            eps = sampler.model.apply_model(x_t, t_tensor, cond)

        x0 = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

        sqrt_alpha_prev = torch.sqrt(torch.tensor(alpha_prev, device=device))
        sqrt_one_minus_alpha_prev = torch.sqrt(torch.tensor(1.0 - alpha_prev, device=device))

        x_t = sqrt_alpha_prev * x0 + sqrt_one_minus_alpha_prev * eps

    return x_t, start_idx


@torch.no_grad()
def ddim_inversion_K_steps(
    sampler,
    x_start: torch.Tensor,
    cond,
    uc,
    cfg_scale: float,
    start_idx: int,
    total_steps: int,
):
    """
    DDIM inversion (re-noising), reversing the same global indices.
    """
    device = x_start.device
    B = x_start.shape[0]
    x_t = x_start

    for global_idx in reversed(range(start_idx, total_steps)):
        t = sampler.ddim_timesteps[global_idx]
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        alpha_t = sampler.ddim_alphas[global_idx]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = sampler.ddim_sqrt_one_minus_alphas[global_idx]

        if global_idx > start_idx:
            alpha_prev = sampler.ddim_alphas[global_idx - 1]
        else:
            alpha_prev = sampler.model.alphas_cumprod[sampler.ddim_timesteps[start_idx]]

        if cfg_scale > 1.0 and uc is not None:
            eps_c = sampler.model.apply_model(x_t, t_tensor, cond)
            eps_u = sampler.model.apply_model(x_t, t_tensor, uc)
            eps = eps_u + cfg_scale * (eps_c - eps_u)
        else:
            eps = sampler.model.apply_model(x_t, t_tensor, cond)

        x0 = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

        sqrt_alpha_prev = torch.sqrt(torch.tensor(alpha_prev, device=device))
        sqrt_one_minus_alpha_prev = torch.sqrt(torch.tensor(1.0 - alpha_prev, device=device))

        x_t = sqrt_alpha_prev * x0 + sqrt_one_minus_alpha_prev * eps

    return x_t


# -----------------------------
# Golden pair generation
# -----------------------------
@torch.no_grad()
def build_golden_pairs(
    model,
    prompts: List[str],
    outdir: str,
    K_steps: int,
    total_ddim_steps: int,
    cfg_forward: float,
    cfg_backward: float,
    batch_size: int,
    height: int,
    width: int,
    frames: int,
    seed: int,
):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)

    C = model.channels
    T = model.temporal_length if frames < 0 else frames
    h, w = height // 8, width // 8
    device = next(model.parameters()).device

    from lvdm.models.samplers.ddim import DDIMSampler
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=total_ddim_steps, ddim_eta=0.0, verbose=False)

    # Resume logic
    existing = [
        int(f.split(".")[0])
        for f in os.listdir(outdir)
        if f.endswith(".pt") and f.split(".")[0].isdigit()
    ]
    start_idx = max(existing) + 1 if existing else 0

    for idx in tqdm(range(start_idx, len(prompts), batch_size), desc="Generating golden pairs"):
        batch_prompts = prompts[idx: idx + batch_size]
        B = len(batch_prompts)

        x_T = torch.randn([B, C, T, h, w], device=device)

        text_emb = model.get_learned_conditioning(batch_prompts)
        cond = {"c_crossattn": [text_emb]}
        uc = {"c_crossattn": [model.get_learned_conditioning([""] * B)]}

        x_mid, start_ddim_idx = ddim_forward_K_steps(
            sampler, x_T, cond, uc, cfg_forward, K_steps
        )

        x_T_target = ddim_inversion_K_steps(
            sampler, x_mid, cond, uc, cfg_backward, start_ddim_idx, total_ddim_steps
        )

        for i in range(B):
            file_idx = idx + i
            torch.save(
                {
                    "x_T": x_T[i].half().cpu(),
                    "x_T_target": x_T_target[i].half().cpu(),
                    "prompt": batch_prompts[i],
                    "meta": {
                        "K_steps": K_steps,
                        "total_ddim_steps": total_ddim_steps,
                        "cfg_forward": cfg_forward,
                        "cfg_backward": cfg_backward,
                        "method": "golden_noise_ddim_corrected",
                    },
                },
                os.path.join(outdir, f"{file_idx:06d}.pt"),
            )

        torch.cuda.empty_cache()


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate golden noise pairs (VideoCrafter + DDIM)")
    parser.add_argument("--seed", type=int, default=20230211)

    parser.add_argument("--videocrafter_root", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=-1)

    parser.add_argument("--K_steps", type=int, default=10)
    parser.add_argument("--total_ddim_steps", type=int, default=50)
    parser.add_argument("--cfg_forward", type=float, default=7.5)
    parser.add_argument("--cfg_backward", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=1)

    args = parser.parse_args()
    seed_everything(args.seed)

    add_repo_to_syspath(args.videocrafter_root)

    from utils.utils import instantiate_from_config
    from scripts.evaluation.funcs import load_model_checkpoint

    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_cfg).cuda()
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    prompts = read_prompts(args.prompt_file)

    build_golden_pairs(
        model=model,
        prompts=prompts,
        outdir=args.outdir,
        K_steps=args.K_steps,
        total_ddim_steps=args.total_ddim_steps,
        cfg_forward=args.cfg_forward,
        cfg_backward=args.cfg_backward,
        batch_size=args.bs,
        height=args.height,
        width=args.width,
        frames=args.frames,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
