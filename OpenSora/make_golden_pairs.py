# make_golden_pairs.py
import os
import sys

# ----------------------------------------------------------------------
# Resolve OpenSora root from environment variable (avoid hard-coded paths)
# ----------------------------------------------------------------------
OPENSORA_ROOT = os.environ.get("OPENSORA_ROOT")
if OPENSORA_ROOT is None:
    raise RuntimeError(
        "OPENSORA_ROOT is not set.\n"
        "Please set it to the path of the OpenSora repository, e.g.:\n"
        "  export OPENSORA_ROOT=/path/to/Open-Sora"
    )
if not os.path.isdir(OPENSORA_ROOT):
    raise RuntimeError(f"OPENSORA_ROOT does not exist or is not a directory: {OPENSORA_ROOT}")

sys.path.insert(0, OPENSORA_ROOT)

# ----------------------------------------------------------------------
# Monkey patch for tensornvme (required for environments without NVMe)
# ----------------------------------------------------------------------
import types

tensornvme = types.ModuleType("tensornvme")
async_file_io = types.ModuleType("async_file_io")
async_file_io.AsyncFileWriter = type("AsyncFileWriter", (), {})
tensornvme.async_file_io = async_file_io

sys.modules["tensornvme"] = tensornvme
sys.modules["tensornvme.async_file_io"] = async_file_io

sys.modules["tensornvme._C"] = types.ModuleType("_C")
sys.modules["tensornvme._C"].Offloader = type("Offloader", (), {})
sys.modules["tensornvme._C"].get_backends = lambda: []

# ----------------------------------------------------------------------
# Standard imports
# ----------------------------------------------------------------------
import argparse
import math
from typing import List

import torch
from colossalai.utils import set_seed

# ----------------------------------------------------------------------
# OpenSora utilities
# ----------------------------------------------------------------------
from opensora.utils.config import parse_configs, parse_alias
from opensora.utils.misc import to_torch_dtype
from opensora.utils.sampling import (
    prepare_models,
    prepare,
    get_noise,
    get_schedule,
    unpack,
    pack,
)
from opensora.utils.inference import prepare_inference_condition


def read_prompts(path: str) -> List[str]:
    """Read prompts from a text file, one prompt per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def compute_latent_num_frames(num_frames: int, temporal_reduction: int, is_causal_vae: bool) -> int:
    """Compute the number of latent frames after temporal downsampling."""
    if num_frames == 1:
        return 1
    if is_causal_vae:
        return (num_frames - 1) // temporal_reduction + 1
    return num_frames // temporal_reduction


def _needs_cond(model) -> bool:
    """Check whether the diffusion model forward() expects 'cond'."""
    import inspect

    try:
        sig = inspect.signature(model.forward)
        return "cond" in sig.parameters
    except Exception:
        return hasattr(model, "cond_in") or hasattr(model, "cond_embedder")


@torch.no_grad()
def rectified_flow_forward_K_steps(
    model,
    model_t5,
    model_clip,
    x_T,
    prompts,
    timesteps,
    K_steps,
    cfg_scale,
    patch_size,
    need_cond,
    is_causal_vae,
    device,
    dtype,
    height,
    width,
):
    """
    Rectified Flow: forward K steps with strong CFG.
    Denoise from x(t=1) to x(t=1 - K*dt).
    """
    bs = len(prompts)
    total_steps = len(timesteps) - 1

    if K_steps > total_steps:
        raise ValueError(f"K_steps={K_steps} > total_steps={total_steps}")

    start_idx = 0
    end_idx = K_steps

    print(f"      Forward: t={timesteps[start_idx]:.4f} -> t={timesteps[end_idx]:.4f}")

    x_t = x_T
    t_lat = x_T.shape[2]

    # 3-batch classifier-free guidance
    prompts_batched = prompts + [""] * bs + [""] * bs
    x_t_batched = torch.cat([x_t, x_t, x_t], dim=0)

    inp = prepare(model_t5, model_clip, x_t_batched, prompt=prompts_batched, patch_size=patch_size)

    extra_kwargs = {}
    if need_cond:
        references = [None] * bs
        masks, masked_ref = prepare_inference_condition(
            x_t, mask_cond="t2v", ref_list=references, causal=is_causal_vae
        )
        cond_5d = torch.cat((masks, masked_ref), dim=1)
        cond_tok = pack(cond_5d, patch_size=patch_size)
        extra_kwargs["cond"] = torch.cat([cond_tok, cond_tok, torch.zeros_like(cond_tok)], dim=0)

    guidance_vec = torch.full((bs * 3,), 1.0, device=device, dtype=dtype)

    for i in range(start_idx, end_idx):
        t_curr = timesteps[i]
        t_prev = timesteps[i + 1]

        t_vec = torch.full((bs * 3,), t_curr, device=device, dtype=dtype)

        pred = model(
            img=inp["img"],
            img_ids=inp["img_ids"],
            txt=inp["txt"],
            txt_ids=inp["txt_ids"],
            timesteps=t_vec,
            y_vec=inp["y_vec"],
            guidance=guidance_vec,
            **extra_kwargs,
        )

        cond_pred, uncond_pred, uncond_2_pred = pred.chunk(3, dim=0)
        pred_cfg = uncond_2_pred + cfg_scale * (cond_pred - uncond_pred)

        # Unpack back to 5D latent for Euler update
        pred_cfg_5d = unpack(pred_cfg, height, width, t_lat, patch_size=patch_size)

        # Euler update
        x_t = x_t + (t_prev - t_curr) * pred_cfg_5d

        # Update packed input
        x_t_batched = torch.cat([x_t, x_t, x_t], dim=0)
        inp["img"] = pack(x_t_batched, patch_size=patch_size)

    return x_t, end_idx


@torch.no_grad()
def rectified_flow_backward_K_steps(
    model,
    model_t5,
    model_clip,
    x_start,
    prompts,
    timesteps,
    start_idx,
    cfg_scale,
    patch_size,
    need_cond,
    is_causal_vae,
    device,
    dtype,
    height,
    width,
):
    """
    Rectified Flow: backward steps with weak CFG.
    Re-noise from x(t=1 - K*dt) back to x(t=1).
    """
    bs = len(prompts)
    t_lat = x_start.shape[2]

    print(f"      Backward: t={timesteps[start_idx]:.4f} -> t={timesteps[0]:.4f}")

    x_t = x_start

    prompts_batched = prompts + [""] * bs + [""] * bs
    x_t_batched = torch.cat([x_t, x_t, x_t], dim=0)

    inp = prepare(model_t5, model_clip, x_t_batched, prompt=prompts_batched, patch_size=patch_size)

    extra_kwargs = {}
    if need_cond:
        references = [None] * bs
        masks, masked_ref = prepare_inference_condition(
            x_t, mask_cond="t2v", ref_list=references, causal=is_causal_vae
        )
        cond_5d = torch.cat((masks, masked_ref), dim=1)
        cond_tok = pack(cond_5d, patch_size=patch_size)
        extra_kwargs["cond"] = torch.cat([cond_tok, cond_tok, torch.zeros_like(cond_tok)], dim=0)

    guidance_vec = torch.full((bs * 3,), 1.0, device=device, dtype=dtype)

    for i in reversed(range(0, start_idx)):
        t_curr = timesteps[i + 1]
        t_prev = timesteps[i]

        t_vec = torch.full((bs * 3,), t_curr, device=device, dtype=dtype)

        pred = model(
            img=inp["img"],
            img_ids=inp["img_ids"],
            txt=inp["txt"],
            txt_ids=inp["txt_ids"],
            timesteps=t_vec,
            y_vec=inp["y_vec"],
            guidance=guidance_vec,
            **extra_kwargs,
        )

        cond_pred, uncond_pred, uncond_2_pred = pred.chunk(3, dim=0)
        pred_cfg = uncond_2_pred + cfg_scale * (cond_pred - uncond_pred)

        pred_cfg_5d = unpack(pred_cfg, height, width, t_lat, patch_size=patch_size)

        x_t = x_t + (t_prev - t_curr) * pred_cfg_5d

        x_t_batched = torch.cat([x_t, x_t, x_t], dim=0)
        inp["img"] = pack(x_t_batched, patch_size=patch_size)

    return x_t


@torch.no_grad()
def build_golden_pairs_opensora(
    model,
    model_ae,
    model_t5,
    model_clip,
    prompts: List[str],
    outdir: str,
    height: int,
    width: int,
    num_frames: int,
    batch_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    # from cfg
    patch_size: int,
    in_channels: int,
    temporal_reduction: int,
    is_causal_vae: bool,
    # golden params
    K_steps: int,
    total_steps: int,
    cfg_forward: float,
    cfg_backward: float,
):
    """Generate and save (x_T, x_T_target) golden pairs for OpenSora2."""
    os.makedirs(outdir, exist_ok=True)

    # Spatial compression factor of the AE (defaults to 16)
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))

    t_lat = compute_latent_num_frames(num_frames, temporal_reduction, is_causal_vae)
    c_lat = in_channels // (patch_size ** 2)
    h_lat = patch_size * math.ceil(height / D)
    w_lat = patch_size * math.ceil(width / D)

    print(f"Video shape: [{num_frames}, {height}, {width}]")
    print(f"Latent x: [B, {c_lat}, {t_lat}, {h_lat}, {w_lat}]")
    print(f"K steps: {K_steps} out of {total_steps}")
    print(f"CFG: forward={cfg_forward}, backward={cfg_backward}")

    need_cond = _needs_cond(model)
    print(f"Model requires cond: {need_cond}")

    image_seq_len = (h_lat // patch_size) * (w_lat // patch_size)
    timesteps = get_schedule(
        num_steps=total_steps,
        image_seq_len=image_seq_len,
        num_frames=t_lat,
        shift=True,
    )

    n = len(prompts)

    # Resume logic: continue from the最大 index already saved
    start_idx = 0
    print(f"Scanning {outdir} for existing pairs...")
    if os.path.exists(outdir):
        existing_files = [f for f in os.listdir(outdir) if f.endswith(".pt")]
        if existing_files:
            existing_indices = [
                int(f.split(".")[0]) for f in existing_files
                if f.split(".")[0].isdigit()
            ]
            if existing_indices:
                start_idx = max(existing_indices) + 1

    if start_idx >= n:
        print(f"  ✓ All {n} pairs already exist. Nothing to do.")
        return
    elif start_idx > 0:
        print(f"  ✓ Resuming from index {start_idx}")
    else:
        print("  No existing pairs found. Starting from 0.")

    print(f"\nGenerating {n - start_idx} remaining pairs...")

    for idx in range(start_idx, n, batch_size):
        batch_prompts = prompts[idx: min(idx + batch_size, n)]
        bs = len(batch_prompts)

        print(f"  [{idx + 1}/{n}] {batch_prompts[0][:50]}...")

        # 1) Sample initial noise x_T
        x_T = get_noise(
            num_samples=bs,
            height=height,
            width=width,
            num_frames=t_lat,
            device=torch.device(device),
            dtype=dtype,
            seed=seed + idx,
            patch_size=patch_size,
            channel=in_channels // (patch_size ** 2),
        )

        # 2) Forward steps (strong CFG)
        print(f"    [1/2] Forward {K_steps} steps (CFG={cfg_forward})")
        x_TminusK, fwd_end_idx = rectified_flow_forward_K_steps(
            model=model,
            model_t5=model_t5,
            model_clip=model_clip,
            x_T=x_T,
            prompts=batch_prompts,
            timesteps=timesteps,
            K_steps=K_steps,
            cfg_scale=cfg_forward,
            patch_size=patch_size,
            need_cond=need_cond,
            is_causal_vae=is_causal_vae,
            device=device,
            dtype=dtype,
            height=height,
            width=width,
        )

        # 3) Backward steps (weak CFG)
        print(f"    [2/2] Backward {K_steps} steps (CFG={cfg_backward})")
        x_T_golden = rectified_flow_backward_K_steps(
            model=model,
            model_t5=model_t5,
            model_clip=model_clip,
            x_start=x_TminusK,
            prompts=batch_prompts,
            timesteps=timesteps,
            start_idx=fwd_end_idx,
            cfg_scale=cfg_backward,
            patch_size=patch_size,
            need_cond=need_cond,
            is_causal_vae=is_causal_vae,
            device=device,
            dtype=dtype,
            height=height,
            width=width,
        )

        # 4) Save pairs
        for i in range(bs):
            file_idx = idx + i
            diff_norm = torch.norm(x_T[i] - x_T_golden[i]).item()

            torch.save(
                {
                    "x_T": x_T[i].half().cpu(),
                    "x_T_target": x_T_golden[i].half().cpu(),
                    "prompt": batch_prompts[i],
                    "meta": {
                        "K_steps": K_steps,
                        "total_steps": total_steps,
                        "cfg_forward": cfg_forward,
                        "cfg_backward": cfg_backward,
                        "method": "golden_noise_opensora",
                        "diff_norm": diff_norm,
                        "patch_size": patch_size,
                        "in_channels": in_channels,
                        "temporal_reduction": temporal_reduction,
                        "is_causal_vae": is_causal_vae,
                        "latent_shape": list(x_T[i].shape),
                        "video_shape": [num_frames, height, width],
                    },
                },
                os.path.join(outdir, f"{file_idx:06d}.pt"),
            )

        print(f"    ✓ Saved (last diff_norm: {diff_norm:.4f})")
        torch.cuda.empty_cache()

    print("\n✓ Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion/inference/256px.py")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=64)

    parser.add_argument("--K_steps", type=int, default=10)
    parser.add_argument("--total_steps", type=int, default=50)
    parser.add_argument("--cfg_forward", type=float, default=7.5)
    parser.add_argument("--cfg_backward", type=float, default=1.0)

    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16")

    args = parser.parse_args()

    # NOTE: OpenSora parse_configs uses sys.argv internally
    sys.argv = ["make_golden_pairs.py", "--config", args.config]
    cfg = parse_configs()
    cfg = parse_alias(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(args.dtype)
    set_seed(args.seed)

    patch_size = cfg.get("patch_size", 2)
    in_channels = cfg["model"]["in_channels"]
    temporal_reduction = cfg.sampling_option.get("temporal_reduction", 1)
    is_causal_vae = cfg.sampling_option.get("is_causal_vae", False)

    print("=" * 60)
    print("Golden Noise (OpenSora2)")
    print("=" * 60)
    print(f"config: {args.config}")
    print(f"device: {device} | dtype: {dtype}")

    print("\n[1/3] Loading models...")
    model, model_ae, model_t5, model_clip, _ = prepare_models(cfg, device, dtype, offload_model=False)
    model.eval()
    print("  ✓ Models loaded")

    print("\n[2/3] Loading prompts...")
    prompts = read_prompts(args.prompt_file)
    print(f"  ✓ Loaded {len(prompts)} prompts")

    print("\n[3/3] Generating golden pairs...")
    build_golden_pairs_opensora(
        model=model,
        model_ae=model_ae,
        model_t5=model_t5,
        model_clip=model_clip,
        prompts=prompts,
        outdir=args.outdir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        batch_size=args.bs,
        dtype=dtype,
        device=device,
        seed=args.seed,
        patch_size=patch_size,
        in_channels=in_channels,
        temporal_reduction=temporal_reduction,
        is_causal_vae=is_causal_vae,
        K_steps=args.K_steps,
        total_steps=args.total_steps,
        cfg_forward=args.cfg_forward,
        cfg_backward=args.cfg_backward,
    )

    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
