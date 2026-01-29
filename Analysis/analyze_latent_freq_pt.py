# analyze_latent_freq_pt.py
# Analyze frequency-domain characteristics in latent space:
#   x      := x_T
#   xg     := x_T_target
#   d      := xg - x
# Metrics:
#   HF_ratio(d): where the *change* lives (high-vs-total power)
#   HF_ratio(x), HF_ratio(xg)
#   Delta_HF := HF_ratio(xg) - HF_ratio(x)

from __future__ import annotations

import os
import argparse
import math
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch


def list_pt_files(folder: str) -> List[str]:
    """List .pt files in a folder, sorted lexicographically."""
    fs = [f for f in os.listdir(folder) if f.endswith(".pt")]
    fs.sort()
    return fs


def quantile_stats(x: torch.Tensor) -> Dict[str, float]:
    """
    Compute summary statistics on a 1D tensor:
    mean, median, p10, p90. Returns NaNs if empty after filtering.
    """
    x = x.reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    x = x.float()
    return {
        "mean": x.mean().item(),
        "median": torch.quantile(x, 0.5).item(),
        "p10": torch.quantile(x, 0.1).item(),
        "p90": torch.quantile(x, 0.9).item(),
    }


@torch.no_grad()
def hf_ratio_from_latent(
    x: torch.Tensor,
    hf_rmin: float = 0.25,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    Compute a simple high-frequency ratio using 2D rFFT over the last two dims.

    Input shapes supported:
      - (H, W)
      - (C, H, W)
      - (C, T, H, W)
      - (B, C, H, W)
      - (B, C, T, H, W)

    We treat the last two dims as (H, W). For each slice over other dims,
    we compute rfft2, power spectrum, then average power across slices and
    compute HF ratio using a radial mask.

    hf_rmin: normalized radial threshold in [0, ~sqrt(2)] (practically use 0.2-0.35).
             0.5 roughly corresponds to Nyquist radius along one axis.
    Returns:
      (hf_ratio, total_power, hf_power)
    """
    xx = x.float()

    # Canonicalize to (B, C, T, H, W)
    if xx.dim() == 2:          # (H, W)
        xx = xx.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,H,W)
    elif xx.dim() == 3:        # (C, H, W) or (T, H, W)
        xx = xx.unsqueeze(0).unsqueeze(2)               # (1, C, 1, H, W)
    elif xx.dim() == 4:        # (B, C, H, W)
        xx = xx.unsqueeze(2)                            # (B, C, 1, H, W)
    elif xx.dim() == 5:        # (B, C, T, H, W)
        pass
    else:
        raise ValueError(f"Unexpected latent shape: {tuple(xx.shape)}")

    B, C, T, H, W = xx.shape

    # rFFT over (H, W)
    X = torch.fft.rfft2(xx, dim=(-2, -1))  # (B, C, T, H, W//2+1)
    P = (X.real * X.real + X.imag * X.imag)  # power

    # Average over slices -> (H, W//2+1)
    P = P.mean(dim=(0, 1, 2))

    # Build radial mask in rFFT grid:
    # ky in [0..H-1], kx in [0..W//2]
    ky = torch.arange(H, device=P.device, dtype=torch.float32)
    kx = torch.arange(W // 2 + 1, device=P.device, dtype=torch.float32)
    yy = ky.view(H, 1).expand(H, W // 2 + 1)
    xx2 = kx.view(1, W // 2 + 1).expand(H, W // 2 + 1)

    # Signed ky magnitude: min(k, H-k)
    yy_signed = torch.minimum(yy, (H - yy))

    # Normalize by Nyquist frequencies
    ry = yy_signed / max(1.0, H / 2.0)
    rx = xx2 / max(1.0, W / 2.0)

    rr = torch.sqrt(ry * ry + rx * rx)
    hf_mask = rr >= hf_rmin

    total_power = P.sum().item()
    hf_power = P[hf_mask].sum().item()
    hf_ratio = hf_power / (total_power + eps)

    return hf_ratio, total_power, hf_power


def _check_matching_files(seed_dirs: List[str], max_files: int) -> List[str]:
    """Ensure all seed folders contain identical .pt filenames (after optional truncation)."""
    files0 = list_pt_files(seed_dirs[0])
    if max_files > 0:
        files0 = files0[:max_files]
    if not files0:
        raise RuntimeError(f"No .pt files found in {seed_dirs[0]}")

    for d in seed_dirs[1:]:
        fs = list_pt_files(d)
        if max_files > 0:
            fs = fs[:max_files]
        if fs != files0:
            set0, set1 = set(files0), set(fs)
            only_in_ref = sorted(list(set0 - set1))[:5]
            only_in_cur = sorted(list(set1 - set0))[:5]
            raise RuntimeError(
                "File mismatch across seed folders.\n"
                f"Reference: {seed_dirs[0]} has {len(files0)} files.\n"
                f"Current:   {d} has {len(fs)} files.\n"
                f"Examples only in reference: {only_in_ref}\n"
                f"Examples only in current:   {only_in_cur}\n"
                "Ensure all seeds produced identical filenames."
            )
    return files0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Parent directory containing seed folders")
    ap.add_argument("--folders", type=str, nargs="+", required=True, help="e.g. golden_pairs0 ... golden_pairs4")
    ap.add_argument("--x_key", type=str, default="x_T")
    ap.add_argument("--target_key", type=str, default="x_T_target")
    ap.add_argument(
        "--hf_rmin",
        type=float,
        default=0.25,
        help="HF threshold radius (normalized). Try 0.20/0.25/0.30 for robustness.",
    )
    ap.add_argument("--max_files", type=int, default=-1, help="Limit number of pt files (-1=all)")
    ap.add_argument("--out_csv", type=str, default="freq_metrics.csv", help="Per-prompt summary CSV")
    ap.add_argument("--out_items_csv", type=str, default="", help="Optional per-item CSV (empty=disable)")
    args = ap.parse_args()

    seed_dirs = [os.path.join(args.root, f) for f in args.folders]
    for d in seed_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing folder: {d}")

    files0 = _check_matching_files(seed_dirs, args.max_files)

    all_hf_d: List[float] = []
    all_hf_x: List[float] = []
    all_hf_xg: List[float] = []
    all_delta_hf: List[float] = []

    # per-prompt values: list of (hf_d, hf_x, hf_xg, delta_hf)
    per_prompt: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)

    # optional per-item rows
    item_rows: List[Dict[str, object]] = []

    for fn in files0:
        for sd in seed_dirs:
            path = os.path.join(sd, fn)
            obj = torch.load(path, map_location="cpu", weights_only=False)

            if args.x_key not in obj or args.target_key not in obj:
                raise KeyError(
                    f"Missing keys in {path}. "
                    f"Need '{args.x_key}' and '{args.target_key}'. Got keys={list(obj.keys())[:40]}"
                )

            x = obj[args.x_key].float()
            xg = obj[args.target_key].float()
            prompt = obj.get("prompt", "")

            d = xg - x

            hf_d, _, _ = hf_ratio_from_latent(d, hf_rmin=args.hf_rmin)
            hf_x, _, _ = hf_ratio_from_latent(x, hf_rmin=args.hf_rmin)
            hf_xg, _, _ = hf_ratio_from_latent(xg, hf_rmin=args.hf_rmin)

            delta_hf = hf_xg - hf_x  # how golden changes HF content of x itself

            all_hf_d.append(hf_d)
            all_hf_x.append(hf_x)
            all_hf_xg.append(hf_xg)
            all_delta_hf.append(delta_hf)

            per_prompt[prompt].append((hf_d, hf_x, hf_xg, delta_hf))

            if args.out_items_csv:
                item_rows.append(
                    {
                        "seed_folder": os.path.basename(sd),
                        "file": fn,
                        "prompt": prompt,
                        "hf_ratio_d": hf_d,
                        "hf_ratio_x": hf_x,
                        "hf_ratio_xg": hf_xg,
                        "delta_hf": delta_hf,
                    }
                )

    # Global stats
    t_hf_d = torch.tensor(all_hf_d, dtype=torch.float32)
    t_hf_x = torch.tensor(all_hf_x, dtype=torch.float32)
    t_hf_xg = torch.tensor(all_hf_xg, dtype=torch.float32)
    t_delta = torch.tensor(all_delta_hf, dtype=torch.float32)

    sd = quantile_stats(t_hf_d)
    sx = quantile_stats(t_hf_x)
    sxg = quantile_stats(t_hf_xg)
    sdel = quantile_stats(t_delta)

    print("\n===== Global HF-ratio summary =====")
    print(
        f"HF_ratio(d=xg-x)  mean={sd['mean']:.6f} median={sd['median']:.6f} "
        f"p10={sd['p10']:.6f} p90={sd['p90']:.6f}"
    )
    print(
        f"HF_ratio(x)       mean={sx['mean']:.6f} median={sx['median']:.6f} "
        f"p10={sx['p10']:.6f} p90={sx['p90']:.6f}"
    )
    print(
        f"HF_ratio(xg)      mean={sxg['mean']:.6f} median={sxg['median']:.6f} "
        f"p10={sxg['p10']:.6f} p90={sxg['p90']:.6f}"
    )
    print(
        f"Delta_HF(xg-x)    mean={sdel['mean']:.6f} median={sdel['median']:.6f} "
        f"p10={sdel['p10']:.6f} p90={sdel['p90']:.6f}"
    )
    print("==================================\n")

    # Write per-prompt summary CSV
    # Sort by mean delta_hf descending (largest HF increase first)
    prompt_items = list(per_prompt.items())
    prompt_items.sort(
        key=lambda kv: float(torch.tensor([v[3] for v in kv[1]]).mean().item()) if kv[1] else -1.0,
        reverse=True,
    )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "prompt",
                "count",
                "hf_d_mean",
                "hf_d_median",
                "hf_d_p10",
                "hf_d_p90",
                "hf_x_mean",
                "hf_xg_mean",
                "delta_hf_mean",
                "delta_hf_median",
            ]
        )

        for prompt, vals in prompt_items:
            vals_t = torch.tensor(vals, dtype=torch.float32)  # (M,4)
            hf_d = vals_t[:, 0]
            hf_x = vals_t[:, 1]
            hf_xg = vals_t[:, 2]
            delt = vals_t[:, 3]

            qd = quantile_stats(hf_d)
            qdel = quantile_stats(delt)

            w.writerow(
                [
                    prompt,
                    int(vals_t.shape[0]),
                    f"{qd['mean']:.8f}",
                    f"{qd['median']:.8f}",
                    f"{qd['p10']:.8f}",
                    f"{qd['p90']:.8f}",
                    f"{hf_x.mean().item():.8f}",
                    f"{hf_xg.mean().item():.8f}",
                    f"{qdel['mean']:.8f}",
                    f"{qdel['median']:.8f}",
                ]
            )

    print(f"Wrote per-prompt HF metrics to: {args.out_csv}")

    # Optional per-item CSV
    if args.out_items_csv:
        with open(args.out_items_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seed_folder", "file", "prompt", "hf_ratio_d", "hf_ratio_x", "hf_ratio_xg", "delta_hf"])
            for r in item_rows:
                w.writerow(
                    [
                        r["seed_folder"],
                        r["file"],
                        r["prompt"],
                        f"{float(r['hf_ratio_d']):.10f}",
                        f"{float(r['hf_ratio_x']):.10f}",
                        f"{float(r['hf_ratio_xg']):.10f}",
                        f"{float(r['delta_hf']):.10f}",
                    ]
                )
        print(f"Wrote per-item HF metrics to: {args.out_items_csv}")

    print("Interpretation:")
    print("- Delta_HF = HF_ratio(xg) - HF_ratio(x). Negative means golden suppresses HF content in x.")
    print("- HF_ratio(d) tells whether the *change itself* is mostly HF or LF.")
    print("- Try --hf_rmin 0.20 / 0.25 / 0.30 to test robustness.")


if __name__ == "__main__":
    main()
