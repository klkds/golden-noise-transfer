"""
Generate weak noise pairs for NPNet training.

For each prompt:
1) Sample x_T ~ N(0, I) in latent space.
2) Pick a diffusion timestep t (high / uniform / fixed).
3) Compute eps_cond and eps_uncond using the model (with cond/uc dicts).
4) Set x_T_target = x_T + lambda_sem * (eps_cond - eps_uncond).
5) Save {x_T, x_T_target, prompt, meta} into .pt.

Design goals:
- No hard-coded paths: VideoCrafter root is provided via --videocrafter_root.
- Optional xformers patch via --patch_xformers.
- Batched generation.
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm


# -----------------------------
# Sys.path helper
# -----------------------------
def add_repo_to_syspath(repo_root: str) -> None:
    repo_root = os.path.abspath(repo_root)
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"Repo root not found: {repo_root}")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# -----------------------------
# Optional xformers patch
# -----------------------------
def maybe_patch_xformers(enable: bool) -> None:
    if not enable:
        return
    try:
        import xformers.ops  # type: ignore

        def safe_memory_efficient_attention(q, k, v, attn_bias=None, op=None):
            d = q.shape[-1]
            scale = (d ** -0.5)
            attn = torch.bmm(q * scale, k.transpose(1, 2))
            attn = torch.softmax(attn, dim=-1)
            out = torch.bmm(attn, v)
            return out

        xformers.ops.memory_efficient_attention = safe_memory_efficient_attention  # type: ignore
        print("[Patch] Using safe_memory_efficient_attention (xformers disabled).")
    except Exception as e:
        print("[Patch] Could not patch xformers, continuing without patch:", e)


# -----------------------------
# Prompt loading
# -----------------------------
def read_prompts(path: str, load_prompts_fn=None) -> List[str]:
    """
    Prefer VideoCrafter's load_prompts if available; fallback to reading text lines.
    """
    if load_prompts_fn is not None:
        try:
            return load_prompts_fn(path)
        except Exception:
            pass

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


# -----------------------------
# Main weak pair builder
# -----------------------------
@torch.no_grad()
def build_weak_pairs(
    model,
    prompts: List[str],
    outdir: str,
    lambda_sem: float = 0.5,
    t_mode: str = "high",
    t_fixed: Optional[int] = None,
    batch_size: int = 4,
    height: int = 512,
    width: int = 512,
    frames: int = -1,
    cfg_scale: float = 1.0,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # VideoCrafter latents are usually H/8, W/8 (VAE factor)
    if (height % 16 != 0) or (width % 16 != 0):
        raise ValueError("height/width should be divisible by 16 for VideoCrafter pipelines.")

    h, w = height // 8, width // 8
    T = model.temporal_length if frames < 0 else frames
    C = model.channels

    device = next(model.parameters()).device
    num_timesteps = int(getattr(model, "num_timesteps", 1000))

    print(f"Latent shape: [{C}, {T}, {h}, {w}]")
    print(f"t_mode={t_mode}, num_timesteps={num_timesteps}, lambda_sem={lambda_sem}, cfg_scale={cfg_scale}")

    def choose_t(bs: int) -> torch.Tensor:
        if t_mode == "high":
            return torch.full((bs,), int(0.9 * num_timesteps), dtype=torch.long, device=device)
        if t_mode == "uniform":
            return torch.randint(int(0.5 * num_timesteps), num_timesteps, (bs,), device=device)
        if t_mode == "fixed":
            if t_fixed is None:
                raise ValueError("t_fixed must be provided when t_mode='fixed'.")
            return torch.full((bs,), int(t_fixed), dtype=torch.long, device=device)
        raise ValueError(f"Unknown t_mode: {t_mode}")

    n = len(prompts)
    n_rounds = (n + batch_size - 1) // batch_size
    print(f"Generating {n} weak pairs in {n_rounds} batches -> {outdir}")

    for ridx in tqdm(range(n_rounds), desc="Batches"):
        s = ridx * batch_size
        e = min(s + batch_size, n)
        batch_prompts = prompts[s:e]
        bs = len(batch_prompts)

        # Sample noise and timestep
        x_T = torch.randn([bs, C, T, h, w], device=device)
        t = choose_t(bs)

        # Build cond and uc in the canonical VideoCrafter format
        text_emb = model.get_learned_conditioning(batch_prompts).to(device)
        cond = {"c_crossattn": [text_emb]}

        uc_emb = model.get_learned_conditioning([""] * bs).to(device)
        uc = {"c_crossattn": [uc_emb]}

        # Predict eps under cond and uc
        eps_cond = model.apply_model(x_T, t, cond)
        eps_uc = model.apply_model(x_T, t, uc)

        # Use the "direction" between cond and uc (optionally with a cfg factor)
        # If cfg_scale=1.0, this is exactly eps_cond - eps_uc.
        d = eps_uc + cfg_scale * (eps_cond - eps_uc) - eps_uc
        # => d = cfg_scale * (eps_cond - eps_uc)

        x_T_target = x_T + float(lambda_sem) * d

        # Save each sample
        for i in range(bs):
            file_idx = s + i
            torch.save(
                {
                    "x_T": x_T[i].half().cpu(),
                    "x_T_target": x_T_target[i].half().cpu(),
                    "prompt": batch_prompts[i],
                    "meta": {
                        "t": int(t[i].item()),
                        "lambda_sem": float(lambda_sem),
                        "cfg_scale": float(cfg_scale),
                        "t_mode": str(t_mode),
                        "latent_shape": [int(C), int(T), int(h), int(w)],
                        "method": "weak_pairs_eps_cond_minus_uncond",
                    },
                },
                os.path.join(outdir, f"{file_idx:06d}.pt"),
            )

        torch.cuda.empty_cache()

    print(f"✓ Done. Saved {n} weak pairs to {outdir}")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate weak pairs for NPNet training (VideoCrafter)")

    p.add_argument("--seed", type=int, default=20230211)

    # VideoCrafter repo root
    p.add_argument(
        "--videocrafter_root",
        type=str,
        required=True,
        help="Path to VideoCrafter repo root (contains configs/, checkpoints/, scripts/, utils/, ...).",
    )

    # Paths relative to videocrafter_root are allowed
    p.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml")
    p.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt")

    p.add_argument("--prompt_file", type=str, required=True, help="Text file with one prompt per line")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for .pt pair files")

    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--frames", type=int, default=-1)

    p.add_argument("--lambda_sem", type=float, default=0.5)
    p.add_argument("--t_mode", type=str, default="high", choices=["high", "uniform", "fixed"])
    p.add_argument("--t_fixed", type=int, default=None)

    p.add_argument("--cfg_scale", type=float, default=1.0, help="Optional CFG scale used in direction computation")
    p.add_argument("--bs", type=int, default=4)

    p.add_argument("--patch_xformers", action="store_true", help="Enable safe attention fallback patch")
    return p


def main() -> None:
    args = get_parser().parse_args()
    seed_everything(args.seed)

    maybe_patch_xformers(args.patch_xformers)

    videocrafter_root = os.path.abspath(args.videocrafter_root)
    add_repo_to_syspath(videocrafter_root)

    from utils.utils import instantiate_from_config  # type: ignore
    from scripts.evaluation.funcs import load_model_checkpoint, load_prompts  # type: ignore

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(videocrafter_root, args.config)
    ckpt_path = args.ckpt_path if os.path.isabs(args.ckpt_path) else os.path.join(videocrafter_root, args.ckpt_path)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

    print("=" * 60)
    print("Weak Pairs Generation (VideoCrafter)")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    cfg = OmegaConf.load(cfg_path)
    model_cfg = cfg.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_cfg).cuda()
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    print("  ✓ Model loaded")

    # Load prompts
    print("\n[2/3] Loading prompts...")
    prompts = read_prompts(args.prompt_file, load_prompts_fn=load_prompts)
    print(f"  ✓ Loaded {len(prompts)} prompts")

    # Build pairs
    print("\n[3/3] Generating weak pairs...")
    build_weak_pairs(
        model=model,
        prompts=prompts,
        outdir=os.path.abspath(args.outdir),
        lambda_sem=args.lambda_sem,
        t_mode=args.t_mode,
        t_fixed=args.t_fixed,
        batch_size=args.bs,
        height=args.height,
        width=args.width,
        frames=args.frames,
        cfg_scale=args.cfg_scale,
    )

    print("=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
