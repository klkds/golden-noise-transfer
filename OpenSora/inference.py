# export OPENSORA_ROOT=/root/autodl-tmp/Open-Sora

import os
import sys

# ----------------------------------------------------------------------
# Resolve OpenSora root from environment variable
# ----------------------------------------------------------------------
OPENSORA_ROOT = os.environ.get("OPENSORA_ROOT")
if OPENSORA_ROOT is None:
    raise RuntimeError(
        "OPENSORA_ROOT is not set.\n"
        "Please set it to the path of the OpenSora repository, e.g.:\n"
        "  export OPENSORA_ROOT=/path/to/Open-Sora"
    )
if not os.path.isdir(OPENSORA_ROOT):
    raise RuntimeError(
        f"OPENSORA_ROOT does not exist or is not a directory: {OPENSORA_ROOT}"
    )

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
import glob
import time
from tqdm import tqdm

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
    get_schedule,
    unpack,
    pack,
)
from opensora.utils.inference import prepare_inference_condition
from opensora.datasets import save_sample


def load_pair(pt_path):
    """Load a .pt file containing weak or golden noise pairs."""
    return torch.load(pt_path, map_location="cpu")


def compute_latent_num_frames(
    num_frames: int, temporal_reduction: int, is_causal_vae: bool
) -> int:
    """Compute the number of latent frames after temporal downsampling."""
    if num_frames == 1:
        return 1
    if is_causal_vae:
        return (num_frames - 1) // temporal_reduction + 1
    return num_frames // temporal_reduction


def _needs_cond(model) -> bool:
    """Check whether the diffusion model requires conditional tokens."""
    import inspect

    try:
        sig = inspect.signature(model.forward)
        return "cond" in sig.parameters
    except Exception:
        return hasattr(model, "cond_in") or hasattr(model, "cond_embedder")


@torch.no_grad()
def opensora_sampling(
    model,
    model_ae,
    model_t5,
    model_clip,
    x_T,
    prompt,
    timesteps,
    guidance,
    patch_size,
    need_cond,
    is_causal_vae,
    height,
    width,
    num_frames,
    device,
    dtype,
):
    """Run OpenSora2 sampling starting from a given latent x_T."""
    bs = x_T.shape[0]
    t_lat = x_T.shape[2]

    x_t = x_T.to(dtype=dtype)

    prompts_batched = prompt + [""] * bs + [""] * bs
    x_t_batched = torch.cat([x_t, x_t, x_t], dim=0)

    inp = prepare(
        model_t5,
        model_clip,
        x_t_batched,
        prompt=prompts_batched,
        patch_size=patch_size,
    )

    extra_kwargs = {}
    if need_cond:
        references = [None] * bs
        masks, masked_ref = prepare_inference_condition(
            x_t, mask_cond="t2v", ref_list=references, causal=is_causal_vae
        )
        cond_5d = torch.cat((masks, masked_ref), dim=1)
        cond_tok = pack(cond_5d, patch_size=patch_size)
        extra_kwargs["cond"] = torch.cat(
            [cond_tok, cond_tok, torch.zeros_like(cond_tok)], dim=0
        )

    guidance_vec = torch.full((bs * 3,), 1.0, device=device, dtype=dtype)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
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
        pred_cfg = uncond_2_pred + guidance * (cond_pred - uncond_pred)

        pred_cfg_5d = unpack(pred_cfg, height, width, t_lat, patch_size=patch_size)

        x_t = x_t + (t_prev - t_curr) * pred_cfg_5d
        x_t_batched = torch.cat([x_t, x_t, x_t], dim=0)
        inp["img"] = pack(x_t_batched, patch_size=patch_size)

    x = model_ae.decode(x_t)
    x = x[:, :, :num_frames]

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diffusion/inference/256px.py",
    )
    parser.add_argument(
        "--pairs_dir",
        type=str,
        required=True,
        help="Directory containing .pt pair files",
    )
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--runtime_bs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16")

    args = parser.parse_args()

    pair_type = "golden" if "golden" in args.pairs_dir else "weak"

    sys.argv = ["inference.py", "--config", args.config]
    cfg = parse_configs()
    cfg = parse_alias(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(args.dtype)
    set_seed(args.seed)

    patch_size = cfg.get("patch_size", 2)
    temporal_reduction = cfg.sampling_option.get("temporal_reduction", 1)
    is_causal_vae = cfg.sampling_option.get("is_causal_vae", False)

    model, model_ae, model_t5, model_clip, _ = prepare_models(
        cfg, device, dtype, offload_model=False
    )
    model.eval()

    need_cond = _needs_cond(model)

    pt_files = sorted(glob.glob(os.path.join(args.pairs_dir, "*.pt")))
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "x_T"), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "x_T_target"), exist_ok=True)

    pbar = tqdm(total=len(pt_files), desc="Progress")
    start_time = time.time()

    for start_idx in range(0, len(pt_files), args.runtime_bs):
        batch_paths = pt_files[start_idx : start_idx + args.runtime_bs]

        x_T_list, x_T_target_list, prompts, metas = [], [], [], []

        for p in batch_paths:
            data = load_pair(p)
            x_T_list.append(data["x_T"].unsqueeze(0).to(device))
            x_T_target_list.append(data["x_T_target"].unsqueeze(0).to(device))
            prompts.append(data["prompt"])
            metas.append(data["meta"])

        x_T = torch.cat(x_T_list, dim=0).to(dtype=dtype)
        x_T_target = torch.cat(x_T_target_list, dim=0).to(dtype=dtype)

        meta0 = metas[0]
        num_frames, height, width = meta0["video_shape"]

        h_lat, w_lat = x_T.shape[3], x_T.shape[4]
        image_seq_len = (h_lat // patch_size) * (w_lat // patch_size)

        timesteps = get_schedule(
            num_steps=args.num_steps,
            image_seq_len=image_seq_len,
            num_frames=x_T.shape[2],
            shift=True,
        )

        samples_x_T = opensora_sampling(
            model,
            model_ae,
            model_t5,
            model_clip,
            x_T,
            prompts,
            timesteps,
            args.guidance,
            patch_size,
            need_cond,
            is_causal_vae,
            height,
            width,
            num_frames,
            device,
            dtype,
        )

        samples_x_T_target = opensora_sampling(
            model,
            model_ae,
            model_t5,
            model_clip,
            x_T_target,
            prompts,
            timesteps,
            args.guidance,
            patch_size,
            need_cond,
            is_causal_vae,
            height,
            width,
            num_frames,
            device,
            dtype,
        )

        for j in range(len(batch_paths)):
            idx = start_idx + j
            fname = f"{idx:06d}"

            save_sample(
                samples_x_T[j].cpu(),
                os.path.join(args.savedir, "x_T", fname),
                fps=args.fps,
            )
            save_sample(
                samples_x_T_target[j].cpu(),
                os.path.join(args.savedir, "x_T_target", fname),
                fps=args.fps,
            )

            with open(os.path.join(args.savedir, f"{fname}.txt"), "w") as f:
                f.write(f"Prompt: {prompts[j]}\n")
                f.write(f"Metadata: {metas[j]}\n")

        pbar.update(len(batch_paths))
        torch.cuda.empty_cache()

    pbar.close()
    elapsed = time.time() - start_time
    print(f"Generated {len(pt_files) * 2} videos in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
