"""
Inference script for visualizing precomputed noise pairs on VideoCrafter.

Given a directory of .pt files that contain:
  - prompt: str
  - x_T: tensor (C,T,H,W) or (1,C,T,H,W)
  - x_T_target: tensor (C,T,H,W) or (1,C,T,H,W)

This script will generate:
  - baseline videos sampled from x_T
  - target   videos sampled from x_T_target

Design goals:
- No hard-coded paths (VideoCrafter repo root and config/ckpt are CLI args)
- Optional xformers attention patch (for certain CUDA/xformers issues)
- Batched sampling
"""

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything


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
    """
    Some environments (certain CUDA + xformers builds) may crash on cutlass kernels.
    This optional patch replaces xformers memory_efficient_attention with a safe PyTorch fallback.
    """
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
# Pair loading
# -----------------------------
def load_pair(pt_path: str) -> Dict[str, Any]:
    # weights_only=False keeps compatibility with dict-based pt files
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def _ensure_5d(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize noise to (B,C,T,H,W).
    Accepts:
      - (C,T,H,W)
      - (1,C,T,H,W)
    """
    if x.dim() == 4:
        return x.unsqueeze(0)
    if x.dim() == 5:
        if x.shape[0] != 1:
            # this script expects single-sample per file; batching happens outside
            raise ValueError(f"Expected B=1 per pt, got shape {tuple(x.shape)}")
        return x
    raise ValueError(f"Unexpected tensor shape {tuple(x.shape)}; expected 4D or 5D.")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate videos from (x_T, x_T_target) pairs using VideoCrafter")

    p.add_argument("--seed", type=int, default=20230211)

    # VideoCrafter repo root (contains scripts/, utils/, configs/, checkpoints/, etc.)
    p.add_argument("--videocrafter_root", type=str, required=True)

    # Allow relative paths w.r.t. videocrafter_root
    p.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml")
    p.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt")

    # pairs + output
    p.add_argument("--pairs_dir", type=str, required=True, help="Directory containing .pt pair files")
    p.add_argument("--savedir", type=str, required=True, help="Output directory for generated videos")
    p.add_argument("--savefps", type=int, default=10)

    # sampling params
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=1.0)
    p.add_argument("--unconditional_guidance_scale", type=float, default=7.5)

    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--runtime_bs", type=int, default=3, help="Batch size for inference")

    p.add_argument("--max_files", type=int, default=-1, help="Limit number of pt files for debugging (-1=all)")

    # optional xformers patch
    p.add_argument("--patch_xformers", action="store_true", help="Enable safe attention fallback patch")

    return p


def main() -> None:
    args = get_parser().parse_args()
    seed_everything(args.seed)

    maybe_patch_xformers(args.patch_xformers)

    videocrafter_root = os.path.abspath(args.videocrafter_root)
    add_repo_to_syspath(videocrafter_root)

    # Import after sys.path update
    from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling  # type: ignore
    from utils.utils import instantiate_from_config  # type: ignore

    # Resolve config/ckpt relative to videocrafter_root
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(videocrafter_root, args.config)
    ckpt_path = args.ckpt_path if os.path.isabs(args.ckpt_path) else os.path.join(videocrafter_root, args.ckpt_path)

    pairs_dir = os.path.abspath(args.pairs_dir)
    savedir = os.path.abspath(args.savedir)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isdir(pairs_dir):
        raise FileNotFoundError(f"pairs_dir not found: {pairs_dir}")

    pair_type = "golden" if "golden" in os.path.basename(pairs_dir).lower() else "weak"

    print("=" * 70)
    print(f"VideoCrafter Inference on Pair Files — {pair_type.upper()} (x_T vs x_T_target)")
    print("=" * 70)

    # -------- 1) Load VideoCrafter model --------
    print("\n[1/3] Loading VideoCrafter model...")
    cfg = OmegaConf.load(cfg_path)
    model_cfg = cfg.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_cfg).cuda()

    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    device = next(model.parameters()).device
    print(f"  ✓ Model loaded on {device}")

    # -------- 2) Load .pt list --------
    print(f"\n[2/3] Scanning pairs_dir: {pairs_dir}")
    pt_files = sorted(glob.glob(os.path.join(pairs_dir, "*.pt")))
    if args.max_files > 0:
        pt_files = pt_files[: args.max_files]
    total_files = len(pt_files)
    if total_files == 0:
        raise RuntimeError(f"No .pt files found in {pairs_dir}")
    print(f"  ✓ Found {total_files} .pt files")

    # Output structure
    out_xT = os.path.join(savedir, "x_T")
    out_xTt = os.path.join(savedir, "x_T_target")
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(out_xT, exist_ok=True)
    os.makedirs(out_xTt, exist_ok=True)

    # -------- 3) Batched inference --------
    print("\n[3/3] Generating videos...")
    start_time = time.time()

    bs = args.runtime_bs
    pbar_all = tqdm(total=total_files, desc="Total", position=0)

    with torch.no_grad():
        for start_idx in range(0, total_files, bs):
            batch_paths = pt_files[start_idx: start_idx + bs]
            cur_bs = len(batch_paths)

            # Load batch
            xT_list: List[torch.Tensor] = []
            xTt_list: List[torch.Tensor] = []
            prompts: List[str] = []
            metas: List[Any] = []

            for pth in batch_paths:
                obj = load_pair(pth)

                if "x_T" not in obj or "x_T_target" not in obj:
                    raise KeyError(f"Missing x_T / x_T_target in {pth}. Keys={list(obj.keys())[:50]}")

                x_T = _ensure_5d(obj["x_T"].float())          # (1,C,T,H,W)
                x_Tt = _ensure_5d(obj["x_T_target"].float())  # (1,C,T,H,W)

                xT_list.append(x_T)
                xTt_list.append(x_Tt)

                prompts.append(obj.get("prompt", ""))
                metas.append(obj.get("meta", {}))

            x_T_batch = torch.cat(xT_list, dim=0).to(device)    # (B,C,T,H,W)
            x_Tt_batch = torch.cat(xTt_list, dim=0).to(device)  # (B,C,T,H,W)

            B, C, T, H, W = x_T_batch.shape
            noise_shape = [B, C, T, H, W]

            # Conditioning
            text_emb = model.get_learned_conditioning(prompts).to(device)
            cond = {"c_crossattn": [text_emb]}

            # Sample from x_T
            samples_xT = batch_ddim_sampling(
                model,
                cond,
                noise_shape,
                n_samples=args.n_samples,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                cfg_scale=args.unconditional_guidance_scale,
                x_T=x_T_batch,
            )

            # Sample from x_T_target
            samples_xTt = batch_ddim_sampling(
                model,
                cond,
                noise_shape,
                n_samples=args.n_samples,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                cfg_scale=args.unconditional_guidance_scale,
                x_T=x_Tt_batch,
            )

            # Save
            filenames = [f"{start_idx + j:06d}" for j in range(cur_bs)]
            save_videos(samples_xT, out_xT, filenames, fps=args.savefps)
            save_videos(samples_xTt, out_xTt, filenames, fps=args.savefps)

            for fname, prompt, meta, src in zip(filenames, prompts, metas, batch_paths):
                with open(os.path.join(savedir, f"{fname}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Meta: {meta}\n")
                    f.write(f"source_pt: {os.path.basename(src)}\n")

            pbar_all.update(cur_bs)
            torch.cuda.empty_cache()

    pbar_all.close()

    elapsed = time.time() - start_time
    print(f"\n✓ Generated {total_files * 2} videos in {elapsed:.2f}s")
    print(f"  Saved to: {savedir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
