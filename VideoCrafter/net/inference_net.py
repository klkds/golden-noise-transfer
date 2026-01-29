# scripts/inference_npnet.py
"""
NPNet inference + baseline comparison on VideoCrafter.

This script:
1) Loads a VideoCrafter T2V model (base) and runs DDIM sampling.
2) Loads NPNetV and a CLIP text encoder (tokenizer + text model).
3) For each saved pair file (.pt), it reads x_T and prompt, then:
   - baseline: sample from x_T
   - ours:     sample from x*_T = NPNetV(x_T, text_embedding)
4) Saves results to disk.

Design:
- No hard-coded paths.
- VideoCrafter repo path is provided via --videocrafter_root and appended to sys.path.
- NPNet code is loaded from --npnet_root WITHOUT polluting global module names (no config collision).

Expected .pt format:
- must contain prompt: str
- must contain noise tensor key: x_T (preferred), or xT, or z_T
"""

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
import importlib.util
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from transformers import CLIPTextModel, CLIPTokenizer


# -----------------------------
# Utilities
# -----------------------------
def add_repo_to_syspath(repo_root: str) -> None:
    repo_root = os.path.abspath(repo_root)
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"Repo root not found: {repo_root}")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_module_from_path(module_name: str, file_path: str):
    """Load a Python module from an explicit file path (prevents name collisions like config.py)."""
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def load_pair(pt_path: str) -> Dict[str, Any]:
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def pick_noise_and_prompt(obj: Dict[str, Any]) -> Tuple[torch.Tensor, str, Any]:
    """
    Supports different key conventions.
    Returns:
      x_T: (C,T,H,W) float tensor on CPU
      prompt: str
      meta: anything (dict preferred)
    """
    prompt = obj.get("prompt", "")
    meta = obj.get("meta", {})

    x = None
    used_key = None
    for k in ["x_T", "xT", "z_T"]:
        if k in obj:
            x = obj[k]
            used_key = k
            break
    if x is None:
        raise KeyError(
            f"Cannot find noise key in pt. Expected one of: x_T, xT, z_T. Keys={list(obj.keys())[:50]}"
        )
    if not torch.is_tensor(x):
        raise TypeError(f"Noise field is not a tensor. key={used_key}, type={type(x)}")

    x = x.float()
    # normalize to (C,T,H,W)
    if x.dim() == 5:
        # (B,C,T,H,W) -> require B=1
        if x.shape[0] != 1:
            raise ValueError(f"Expected single sample per pt, got shape {tuple(x.shape)}")
        x = x[0]
    if x.dim() != 4:
        raise ValueError(f"Expected noise tensor of shape (C,T,H,W), got {tuple(x.shape)}")

    return x, prompt, meta


@torch.no_grad()
def clip_mean_pool(
    text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, prompts: List[str], device: torch.device
) -> torch.Tensor:
    """
    CLIP text embeddings with attention-mask mean pooling (aligned with NPNet training).
    Returns:
      E_txt: (B, D)
    """
    inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = text_encoder(input_ids=inputs["input_ids"])
    hidden = out.last_hidden_state.float()  # (B, L, D)

    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, D)
    return pooled


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Argument parsing
# -----------------------------
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NPNet inference on VideoCrafter (baseline vs ours)")

    p.add_argument("--seed", type=int, default=20241116)

    # Path to VideoCrafter repo root
    p.add_argument(
        "--videocrafter_root",
        type=str,
        required=True,
        help="Path to the VideoCrafter repo root (contains scripts/, utils/, configs/, etc.)",
    )

    # Path to NPNet repo root (contains model.py and config.py)
    p.add_argument(
        "--npnet_root",
        type=str,
        required=True,
        help="Path to NPNet repo root (contains model.py and config.py).",
    )

    # T2V base model config + checkpoint (paths relative to videocrafter_root allowed)
    p.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml")
    p.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt")

    # NPNet checkpoint (path relative to npnet_root allowed)
    p.add_argument("--npnet_ckpt_path", type=str, default="npnet_final.pth")

    # CLIP local model folder (tokenizer + text encoder)
    p.add_argument("--clip_path", type=str, required=True, help="Local path to CLIP tokenizer & text encoder folder")

    # pairs dir and output dir
    p.add_argument("--pairs_dir", type=str, required=True, help="Directory containing .pt files")
    p.add_argument("--savedir", type=str, required=True, help="Directory to save generated videos")

    # sampling params
    p.add_argument("--savefps", type=int, default=10)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=1.0)
    p.add_argument("--unconditional_guidance_scale", type=float, default=7.5)

    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--runtime_bs", type=int, default=1)

    p.add_argument("--max_files", type=int, default=-1, help="Limit number of pt files for quick debugging (-1=all)")

    return p


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = get_parser().parse_args()
    seed_everything(args.seed)

    videocrafter_root = os.path.abspath(args.videocrafter_root)
    npnet_root = os.path.abspath(args.npnet_root)

    # Put VideoCrafter on sys.path (so we can import scripts/utils)
    add_repo_to_syspath(videocrafter_root)

    # Import after sys.path update (VideoCrafter)
    from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling  # type: ignore
    from utils.utils import instantiate_from_config  # type: ignore

    # Load NPNet modules by explicit paths to avoid config.py name collisions
    npnet_config = load_module_from_path("npnet_config", os.path.join(npnet_root, "config.py"))
    npnet_model_mod = load_module_from_path("npnet_model", os.path.join(npnet_root, "model.py"))
    NPNetV = getattr(npnet_model_mod, "NPNetV")

    print("=" * 70)
    print("NPNet Inference on VideoCrafter — Baseline vs Ours")
    print("=" * 70)

    # Resolve paths (allow relative to VideoCrafter/NPNet roots)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(videocrafter_root, args.config)
    ckpt_path = args.ckpt_path if os.path.isabs(args.ckpt_path) else os.path.join(videocrafter_root, args.ckpt_path)
    npnet_ckpt_path = (
        args.npnet_ckpt_path if os.path.isabs(args.npnet_ckpt_path) else os.path.join(npnet_root, args.npnet_ckpt_path)
    )

    clip_path = os.path.abspath(args.clip_path)
    pairs_dir = os.path.abspath(args.pairs_dir)
    savedir = os.path.abspath(args.savedir)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"T2V config not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"T2V checkpoint not found: {ckpt_path}")
    if not os.path.exists(npnet_ckpt_path):
        raise FileNotFoundError(f"NPNet checkpoint not found: {npnet_ckpt_path}")
    if not os.path.isdir(clip_path):
        raise FileNotFoundError(f"CLIP folder not found: {clip_path}")
    if not os.path.isdir(pairs_dir):
        raise FileNotFoundError(f"pairs_dir not found: {pairs_dir}")

    # -------- 1) Load T2V model --------
    print("\n[1/4] Loading VideoCrafter T2V model...")
    t2v_config = OmegaConf.load(cfg_path)
    model_config = t2v_config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config).cuda()
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    device = next(model.parameters()).device
    print(f"  ✓ T2V loaded on {device}")

    # -------- 2) Load NPNet + CLIP --------
    print("\n[2/4] Loading NPNet + CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    text_encoder = CLIPTextModel.from_pretrained(clip_path, torch_dtype=torch.float32).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    npnet = NPNetV(
        channels=int(npnet_config.CHANNELS),
        T=int(npnet_config.TEMPORAL_DIM),
        H=int(npnet_config.HEIGHT),
        W=int(npnet_config.WIDTH),
        freq_decay=float(npnet_config.FREQ_DECAY),
    ).to(device)

    state = torch.load(npnet_ckpt_path, map_location="cpu", weights_only=False)
    npnet.load_state_dict(state, strict=True)
    npnet.eval()
    print(f"  ✓ NPNet loaded: {npnet_ckpt_path}")

    # -------- 3) Load pairs --------
    print("\n[3/4] Loading pair files...")
    pt_files = sorted(glob.glob(os.path.join(pairs_dir, "*.pt")))
    if args.max_files > 0:
        pt_files = pt_files[: args.max_files]
    total_files = len(pt_files)
    if total_files == 0:
        raise RuntimeError(f"No .pt files found in {pairs_dir}")
    print(f"  ✓ Found {total_files} pt files")

    # Output dirs
    ensure_dir(savedir)
    out_baseline = os.path.join(savedir, "baseline_xT")
    out_npnet = os.path.join(savedir, "npnet_x_star_T")
    ensure_dir(out_baseline)
    ensure_dir(out_npnet)

    # -------- 4) Batched generation --------
    print("\n[4/4] Generating videos...")
    start_time = time.time()

    runtime_bs = int(args.runtime_bs)
    pbar = tqdm(total=total_files, desc="Total")

    with torch.no_grad():
        for start_idx in range(0, total_files, runtime_bs):
            batch_paths = pt_files[start_idx : start_idx + runtime_bs]
            cur_bs = len(batch_paths)

            xT_list: List[torch.Tensor] = []
            prompts: List[str] = []
            metas: List[Any] = []
            sources: List[str] = []

            for pth in batch_paths:
                obj = load_pair(pth)
                x_T, prompt, meta = pick_noise_and_prompt(obj)
                xT_list.append(x_T.unsqueeze(0))  # (1,C,T,H,W)
                prompts.append(prompt)
                metas.append(meta)
                sources.append(os.path.basename(pth))

            x_T_batch = torch.cat(xT_list, dim=0).to(device)  # (B,C,T,H,W)
            B, C, T, H, W = x_T_batch.shape
            noise_shape = [B, C, T, H, W]

            # NPNet: x*_T
            E_txt = clip_mean_pool(text_encoder, tokenizer, prompts, device=device)  # (B,D)
            x_star_T_batch = npnet(x_T_batch, E_txt)  # (B,C,T,H,W)

            # VideoCrafter conditioning
            text_emb = model.get_learned_conditioning(prompts).to(device)
            cond = {"c_crossattn": [text_emb]}

            # baseline sampling
            samples_xT = batch_ddim_sampling(
                model,
                cond,
                noise_shape,
                n_samples=int(args.n_samples),
                ddim_steps=int(args.ddim_steps),
                ddim_eta=float(args.ddim_eta),
                cfg_scale=float(args.unconditional_guidance_scale),
                x_T=x_T_batch,
            )

            # ours sampling
            samples_x_star_T = batch_ddim_sampling(
                model,
                cond,
                noise_shape,
                n_samples=int(args.n_samples),
                ddim_steps=int(args.ddim_steps),
                ddim_eta=float(args.ddim_eta),
                cfg_scale=float(args.unconditional_guidance_scale),
                x_T=x_star_T_batch,
            )

            # save
            filenames = [f"{start_idx + j:06d}" for j in range(cur_bs)]
            save_videos(samples_xT, out_baseline, filenames, fps=int(args.savefps))
            save_videos(samples_x_star_T, out_npnet, filenames, fps=int(args.savefps))

            for fname, prompt, meta, src in zip(filenames, prompts, metas, sources):
                with open(os.path.join(savedir, f"{fname}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Meta: {meta}\n")
                    f.write(f"source_pt: {src}\n")

            pbar.update(cur_bs)
            torch.cuda.empty_cache()

    pbar.close()
    elapsed = time.time() - start_time
    print(f"\n✓ Generated {total_files * 2} videos in {elapsed:.2f}s")
    print(f"  Saved to: {savedir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
