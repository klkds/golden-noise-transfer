# analyze_spatiotemporal_flicker_pt.py
# Analyze spatiotemporal "flicker" signatures in latent space.
# x  := x_T
# xg := x_T_target
# d  := xg - x
#
# Spatial metric:
#   sp_hf_ratio(*) : 2D rFFT over (H,W) averaged across slices -> HF/total power.
#
# Temporal metrics:
#   t_hf_ratio(*)  : 1D rFFT over T averaged across (C,H,W) -> HF/total power (excluding DC).
#   tDiffRMS(d)    : RMS of frame-to-frame difference in time.
#   tDiffRel(d)    : tDiffRMS(d) / RMS(d) (normalized), robust scale-free flicker indicator.

from __future__ import annotations

import os
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch


def list_pt_files(folder: str) -> List[str]:
    fs = [f for f in os.listdir(folder) if f.endswith(".pt")]
    fs.sort()
    return fs


def quantile_stats(x: torch.Tensor) -> Dict[str, float]:
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


def _check_matching_files(seed_dirs: List[str], max_files: int) -> List[str]:
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
                "Ensure all seeds produced identical filenames per seed."
            )
    return files0


def _to_CTHW(x: torch.Tensor, assume_3d_is: str = "CHW") -> torch.Tensor:
    """
    Convert x to (C, T, H, W).
    Supported:
      - (C,T,H,W) -> unchanged
      - (C,H,W)   -> add T=1
      - (H,W)     -> treat as C=1,T=1
      - (T,H,W)   -> if assume_3d_is == "THW", interpret as time-major and add C=1
    """
    t = x.float()
    if t.dim() == 4:
        return t
    if t.dim() == 3:
        if assume_3d_is.upper() == "THW":
            # (T,H,W) -> (1,T,H,W)
            return t.unsqueeze(0)
        # default: (C,H,W) -> (C,1,H,W)
        return t.unsqueeze(1)
    if t.dim() == 2:
        # (H,W) -> (1,1,H,W)
        return t.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")


@torch.no_grad()
def spatial_hf_ratio(x_cthw: torch.Tensor, hf_rmin: float = 0.25, eps: float = 1e-12) -> float:
    """
    x_cthw: (C,T,H,W)
    Compute 2D rFFT over (H,W) per slice, average power over (C,T), then HF ratio by radial mask.
    """
    t = x_cthw.float()
    C, T, H, W = t.shape

    X = torch.fft.rfft2(t, dim=(-2, -1))  # (C,T,H,W//2+1)
    P = (X.real * X.real + X.imag * X.imag).mean(dim=(0, 1))  # (H, W//2+1)

    ky = torch.arange(H, device=P.device, dtype=torch.float32)
    kx = torch.arange(W // 2 + 1, device=P.device, dtype=torch.float32)
    yy = ky.view(H, 1).expand(H, W // 2 + 1)
    xx = kx.view(1, W // 2 + 1).expand(H, W // 2 + 1)

    yy_signed = torch.minimum(yy, (H - yy))
    ry = yy_signed / max(1.0, H / 2.0)
    rx = xx / max(1.0, W / 2.0)
    rr = torch.sqrt(ry * ry + rx * rx)

    hf_mask = rr >= hf_rmin
    total = P.sum().item()
    hf = P[hf_mask].sum().item()
    return hf / (total + eps)


@torch.no_grad()
def temporal_hf_ratio(x_cthw: torch.Tensor, hf_rmin: float = 0.25, eps: float = 1e-12) -> float:
    """
    x_cthw: (C,T,H,W)
    Compute 1D rFFT over T for each (C,H,W), average power, then HF ratio.
    hf_rmin is normalized relative to Nyquist (0..1), where 1 means the Nyquist bin.
    DC is excluded from both numerator and denominator for robustness.
    """
    t = x_cthw.float()
    C, T, H, W = t.shape
    if T < 2:
        return float("nan")

    X = torch.fft.rfft(t, dim=1)  # (C, T//2+1, H, W)
    P = (X.real * X.real + X.imag * X.imag).mean(dim=(0, 2, 3))  # (K,)

    K = P.numel()
    if K <= 2:
        return float("nan")

    idx = torch.arange(K, device=P.device, dtype=torch.float32)
    fr = idx / (K - 1)  # [0..1]
    hf_mask = (idx > 0) & (fr >= hf_rmin)  # exclude DC
    total = P[1:].sum().item()
    hf = P[hf_mask].sum().item()
    return hf / (total + eps)


@torch.no_grad()
def temporal_diff_metrics(x_cthw: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float]:
    """
    x_cthw: (C,T,H,W)
    Returns:
      t_diff_rms: RMS of (x[:,t+1]-x[:,t]) over all elements and t
      t_diff_rel: t_diff_rms / RMS(x) (scale-free)
    """
    t = x_cthw.float()
    C, T, H, W = t.shape
    if T < 2:
        return float("nan"), float("nan")

    dx = t[:, 1:] - t[:, :-1]  # (C,T-1,H,W)
    t_diff_rms = torch.sqrt(torch.mean(dx * dx)).item()
    base_rms = torch.sqrt(torch.mean(t * t)).item()
    t_diff_rel = t_diff_rms / (base_rms + eps)
    return t_diff_rms, t_diff_rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--folders", type=str, nargs="+", required=True, help="e.g. golden_pairs0 ... golden_pairs4")
    ap.add_argument("--x_key", type=str, default="x_T")
    ap.add_argument("--target_key", type=str, default="x_T_target")

    ap.add_argument("--sp_hf_rmin", type=float, default=0.25, help="Spatial HF threshold radius (normalized)")
    ap.add_argument("--t_hf_rmin", type=float, default=0.25, help="Temporal HF threshold (normalized to Nyquist)")

    ap.add_argument(
        "--assume_3d_is",
        type=str,
        default="CHW",
        choices=["CHW", "THW"],
        help="How to interpret 3D tensors: CHW (default) or THW (treat first dim as time).",
    )

    ap.add_argument("--max_files", type=int, default=-1)
    ap.add_argument("--out_csv", type=str, default="st_flicker_metrics.csv", help="Per-prompt summary CSV")
    ap.add_argument("--out_items_csv", type=str, default="", help="Optional per-item CSV (empty=disable)")
    args = ap.parse_args()

    seed_dirs = [os.path.join(args.root, f) for f in args.folders]
    for d in seed_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing folder: {d}")

    files0 = _check_matching_files(seed_dirs, args.max_files)

    # per-prompt values: list of (sp_hf_d, t_hf_d, tdiff_rms_d, tdiff_rel_d)
    per_prompt: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)

    # global lists (may include NaNs for temporal metrics if T<2)
    g_sp_hf_d, g_t_hf_d, g_tdiff_rms_d, g_tdiff_rel_d = [], [], [], []
    g_t_hf_x, g_t_hf_xg, g_sp_hf_x, g_sp_hf_xg = [], [], [], []

    item_rows: List[Dict[str, object]] = []

    for i, fn in enumerate(files0):
        for sd in seed_dirs:
            path = os.path.join(sd, fn)
            obj = torch.load(path, map_location="cpu", weights_only=False)

            if args.x_key not in obj or args.target_key not in obj:
                raise KeyError(
                    f"Missing keys in {path}. Need '{args.x_key}' and '{args.target_key}'. "
                    f"Got keys={list(obj.keys())[:40]}"
                )

            x_raw = obj[args.x_key]
            xg_raw = obj[args.target_key]
            prompt = obj.get("prompt", "")

            x = _to_CTHW(x_raw, assume_3d_is=args.assume_3d_is)
            xg = _to_CTHW(xg_raw, assume_3d_is=args.assume_3d_is)

            if x.shape != xg.shape:
                raise RuntimeError(f"Shape mismatch in {path}: x={tuple(x.shape)} vs xg={tuple(xg.shape)}")

            d = xg - x  # (C,T,H,W)

            sp_hf_d = spatial_hf_ratio(d, hf_rmin=args.sp_hf_rmin)
            sp_hf_x = spatial_hf_ratio(x, hf_rmin=args.sp_hf_rmin)
            sp_hf_xg = spatial_hf_ratio(xg, hf_rmin=args.sp_hf_rmin)

            t_hf_d = temporal_hf_ratio(d, hf_rmin=args.t_hf_rmin)
            t_hf_x = temporal_hf_ratio(x, hf_rmin=args.t_hf_rmin)
            t_hf_xg = temporal_hf_ratio(xg, hf_rmin=args.t_hf_rmin)

            tdiff_rms_d, tdiff_rel_d = temporal_diff_metrics(d)

            per_prompt[prompt].append((sp_hf_d, t_hf_d, tdiff_rms_d, tdiff_rel_d))

            g_sp_hf_d.append(sp_hf_d)
            g_t_hf_d.append(t_hf_d)
            g_tdiff_rms_d.append(tdiff_rms_d)
            g_tdiff_rel_d.append(tdiff_rel_d)
            g_sp_hf_x.append(sp_hf_x)
            g_sp_hf_xg.append(sp_hf_xg)
            g_t_hf_x.append(t_hf_x)
            g_t_hf_xg.append(t_hf_xg)

            if args.out_items_csv:
                item_rows.append(
                    {
                        "seed_folder": os.path.basename(sd),
                        "file": fn,
                        "prompt": prompt,
                        "sp_hf_ratio_d": sp_hf_d,
                        "t_hf_ratio_d": t_hf_d,
                        "t_diff_rms_d": tdiff_rms_d,
                        "t_diff_rel_d": tdiff_rel_d,
                        "sp_hf_ratio_x": sp_hf_x,
                        "sp_hf_ratio_xg": sp_hf_xg,
                        "t_hf_ratio_x": t_hf_x,
                        "t_hf_ratio_xg": t_hf_xg,
                        "shape_CTHW": str(tuple(x.shape)),
                    }
                )

        if (i + 1) % 200 == 0 or (i + 1) == len(files0):
            print(f"processed {i + 1}/{len(files0)} files")

    def summary_line(name: str, arr: List[float]) -> None:
        t = torch.tensor(arr, dtype=torch.float32)
        q = quantile_stats(t)
        print(f"{name:14s} mean={q['mean']:.6f} median={q['median']:.6f} p10={q['p10']:.6f} p90={q['p90']:.6f}")

    print("\n===== Global spatiotemporal summary (all prompts × seeds) =====")
    summary_line("sp_hf(d)", g_sp_hf_d)
    summary_line("t_hf(d)", g_t_hf_d)
    summary_line("tDiffRMS(d)", g_tdiff_rms_d)
    summary_line("tDiffRel(d)", g_tdiff_rel_d)
    print("---- reference (x / xg) ----")
    summary_line("sp_hf(x)", g_sp_hf_x)
    summary_line("sp_hf(xg)", g_sp_hf_xg)
    summary_line("t_hf(x)", g_t_hf_x)
    summary_line("t_hf(xg)", g_t_hf_xg)
    print("==============================================================\n")

    # Write per-prompt summary CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "prompt",
                "count",
                "sp_hf_d_mean",
                "sp_hf_d_median",
                "t_hf_d_mean",
                "t_hf_d_median",
                "tDiffRel_d_mean",
                "tDiffRel_d_median",
            ]
        )
        for p, vals in per_prompt.items():
            vals_t = torch.tensor(vals, dtype=torch.float32)  # (M,4)
            sp = vals_t[:, 0]
            th = vals_t[:, 1]
            tr = vals_t[:, 3]

            qsp = quantile_stats(sp)
            qth = quantile_stats(th)
            qtr = quantile_stats(tr)

            w.writerow(
                [
                    p,
                    int(vals_t.shape[0]),
                    f"{qsp['mean']:.8f}",
                    f"{qsp['median']:.8f}",
                    f"{qth['mean']:.8f}",
                    f"{qth['median']:.8f}",
                    f"{qtr['mean']:.8f}",
                    f"{qtr['median']:.8f}",
                ]
            )

    print(f"Saved per-prompt summary to: {args.out_csv}")

    # Optional per-item CSV
    if args.out_items_csv:
        with open(args.out_items_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "seed_folder",
                    "file",
                    "prompt",
                    "sp_hf_ratio_d",
                    "t_hf_ratio_d",
                    "t_diff_rms_d",
                    "t_diff_rel_d",
                    "sp_hf_ratio_x",
                    "sp_hf_ratio_xg",
                    "t_hf_ratio_x",
                    "t_hf_ratio_xg",
                    "shape_CTHW",
                ]
            )
            for r in item_rows:
                w.writerow(
                    [
                        r["seed_folder"],
                        r["file"],
                        r["prompt"],
                        f"{float(r['sp_hf_ratio_d']):.10f}",
                        f"{float(r['t_hf_ratio_d']):.10f}",
                        f"{float(r['t_diff_rms_d']):.10f}",
                        f"{float(r['t_diff_rel_d']):.10f}",
                        f"{float(r['sp_hf_ratio_x']):.10f}",
                        f"{float(r['sp_hf_ratio_xg']):.10f}",
                        f"{float(r['t_hf_ratio_x']):.10f}",
                        f"{float(r['t_hf_ratio_xg']):.10f}",
                        r["shape_CTHW"],
                    ]
                )
        print(f"Saved per-item metrics to: {args.out_items_csv}")

    print("How to interpret:")
    print("- If sp_hf(d) is LOW but t_hf(d) / tDiffRel(d) is HIGH, the displacement is spatially smooth")
    print("  but temporally jittery -> a flicker-like signature in latent displacement.")
    print("- Try --t_hf_rmin 0.30 / 0.40 to verify robustness against the HF threshold.")


if __name__ == "__main__":
    main()
