# golden_normal_diff.py
# normal := x_T (identity). golden := x_T_target.
# This script measures the latent-space difference magnitude between golden and normal,
# not stability (no DirStab/CV metrics).

from __future__ import annotations

import os
import argparse
import math
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import torch


def list_pt_files(folder: str) -> List[str]:
    """List .pt files in a folder, sorted lexicographically."""
    fs = [f for f in os.listdir(folder) if f.endswith(".pt")]
    fs.sort()
    return fs


def quantile_stats(x: torch.Tensor) -> Dict[str, float]:
    """
    Compute robust summary statistics on a 1D tensor.
    Returns mean/median/p10/p90 (NaN if empty after filtering).
    """
    x = x.reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}

    x = x.float()
    mean = x.mean().item()
    median = torch.quantile(x, 0.5).item()
    p10 = torch.quantile(x, 0.1).item()
    p90 = torch.quantile(x, 0.9).item()
    return {"mean": mean, "median": median, "p10": p10, "p90": p90}


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity between two tensors (flattened)."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    an = torch.linalg.vector_norm(a).item()
    bn = torch.linalg.vector_norm(b).item()
    if an < eps or bn < eps:
        return float("nan")
    return (a @ b).item() / (an * bn + eps)


def load_metrics(
    pt_path: str,
    x_key: str = "x_T",
    target_key: str = "x_T_target",
) -> Tuple[str, float, float, float, float, float]:
    """
    Load a pair file and compute:
      - diff_l2  = ||xg - x||_2
      - x_l2     = ||x||_2
      - rel_l2   = ||xg - x||_2 / ||x||_2
      - rms      = sqrt(mean((xg - x)^2)) (per-dim RMS)
      - cos      = cosine(x, xg)
    """
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if x_key not in obj:
        raise KeyError(f"Missing key '{x_key}' in {pt_path}")
    if target_key not in obj:
        raise KeyError(f"Missing key '{target_key}' in {pt_path}")

    x = obj[x_key].float()
    xg = obj[target_key].float()
    prompt = obj.get("prompt", "")

    dx = (xg - x).reshape(-1)

    diff_l2 = torch.linalg.vector_norm(dx).item()
    x_l2 = torch.linalg.vector_norm(x.reshape(-1)).item()
    rel_l2 = diff_l2 / max(x_l2, 1e-12)
    rms = torch.sqrt(torch.mean(dx * dx)).item()
    cos_x_xg = cosine(x, xg)

    return prompt, diff_l2, x_l2, rel_l2, rms, cos_x_xg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Parent directory containing seed folders")
    ap.add_argument(
        "--folders",
        type=str,
        nargs="+",
        required=True,
        help="Seed folders (e.g., golden_pairs0 golden_pairs1 ...)",
    )
    ap.add_argument("--x_key", type=str, default="x_T")
    ap.add_argument("--target_key", type=str, default="x_T_target")
    ap.add_argument("--max_files", type=int, default=-1, help="Limit number of pt files for quick test (-1=all)")
    ap.add_argument("--out_csv", type=str, default="golden_vs_normal_magnitude.csv", help="Per-prompt summary CSV")
    ap.add_argument("--out_items_csv", type=str, default="", help="Optional per-item CSV (empty=disable)")
    args = ap.parse_args()

    seed_dirs = [os.path.join(args.root, f) for f in args.folders]
    for d in seed_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing folder: {d}")

    # Use the first folder as filename reference
    files0 = list_pt_files(seed_dirs[0])
    if args.max_files > 0:
        files0 = files0[: args.max_files]
    if not files0:
        raise RuntimeError(f"No .pt files found in {seed_dirs[0]}")

    # Verify matching filenames across seed folders
    for d in seed_dirs[1:]:
        fs = list_pt_files(d)
        if args.max_files > 0:
            fs = fs[: args.max_files]
        if fs != files0:
            set0, set1 = set(files0), set(fs)
            only_in_ref = sorted(list(set0 - set1))[:5]
            only_in_cur = sorted(list(set1 - set0))[:5]
            raise RuntimeError(
                "File mismatch across seed folders.\n"
                f"Reference folder: {seed_dirs[0]} has {len(files0)} files.\n"
                f"Current folder:   {d} has {len(fs)} files.\n"
                f"Examples only in reference: {only_in_ref}\n"
                f"Examples only in current:   {only_in_cur}\n"
                "Ensure all seeds produced identical filenames."
            )

    S = len(seed_dirs)
    N = len(files0)
    print(f"Found {N} files across {S} seeds. Computing golden-vs-normal (target - normal) magnitude...")

    # Aggregate by prompt: each prompt gets S samples (one per seed folder)
    by_prompt: Dict[str, List[Tuple[float, float, float, float, float]]] = defaultdict(list)

    # Optional per-item rows (for outlier inspection)
    per_item_rows: List[Dict[str, object]] = []

    for i, fn in enumerate(files0):
        for d in seed_dirs:
            path = os.path.join(d, fn)
            prompt, diff_l2, x_l2, rel_l2, rms, cos_x_xg = load_metrics(
                path, x_key=args.x_key, target_key=args.target_key
            )
            by_prompt[prompt].append((diff_l2, x_l2, rel_l2, rms, cos_x_xg))

            if args.out_items_csv:
                per_item_rows.append(
                    {
                        "file": fn,
                        "seed_folder": os.path.basename(d),
                        "prompt": prompt,
                        "diff_l2": diff_l2,
                        "x_l2": x_l2,
                        "rel_l2": rel_l2,
                        "rms": rms,
                        "cos_x_xg": cos_x_xg,
                    }
                )

        if (i + 1) % 200 == 0 or (i + 1) == N:
            print(f"  processed {i + 1}/{N} files")

    # Per-prompt summary rows
    prompt_rows: List[Dict[str, object]] = []
    all_diff, all_x, all_rel, all_rms, all_cos = [], [], [], [], []

    for p, vals in by_prompt.items():
        diff = torch.tensor([v[0] for v in vals], dtype=torch.float32)
        x = torch.tensor([v[1] for v in vals], dtype=torch.float32)
        rel = torch.tensor([v[2] for v in vals], dtype=torch.float32)
        rms = torch.tensor([v[3] for v in vals], dtype=torch.float32)
        cosv = torch.tensor([v[4] for v in vals], dtype=torch.float32)

        sd = quantile_stats(diff)
        sx = quantile_stats(x)
        sr = quantile_stats(rel)
        ss = quantile_stats(rms)
        sc = quantile_stats(cosv)

        prompt_rows.append(
            {
                "prompt": p,
                "count": int(diff.numel()),
                "diff_l2_mean": sd["mean"],
                "diff_l2_median": sd["median"],
                "diff_l2_p10": sd["p10"],
                "diff_l2_p90": sd["p90"],
                "x_l2_mean": sx["mean"],
                "rel_l2_mean": sr["mean"],
                "rel_l2_median": sr["median"],
                "rel_l2_p10": sr["p10"],
                "rel_l2_p90": sr["p90"],
                "rms_mean": ss["mean"],
                "rms_median": ss["median"],
                "cos_mean": sc["mean"],
                "cos_median": sc["median"],
            }
        )

        all_diff.append(diff)
        all_x.append(x)
        all_rel.append(rel)
        all_rms.append(rms)
        all_cos.append(cosv)

    # Global summaries
    all_diff_t = torch.cat(all_diff) if all_diff else torch.tensor([], dtype=torch.float32)
    all_x_t = torch.cat(all_x) if all_x else torch.tensor([], dtype=torch.float32)
    all_rel_t = torch.cat(all_rel) if all_rel else torch.tensor([], dtype=torch.float32)
    all_rms_t = torch.cat(all_rms) if all_rms else torch.tensor([], dtype=torch.float32)
    all_cos_t = torch.cat(all_cos) if all_cos else torch.tensor([], dtype=torch.float32)

    g_diff = quantile_stats(all_diff_t)
    g_x = quantile_stats(all_x_t)
    g_rel = quantile_stats(all_rel_t)
    g_rms = quantile_stats(all_rms_t)
    g_cos = quantile_stats(all_cos_t)

    print("\n===== Global magnitude summary (all prompts × seeds) =====")
    print(f"||x||               mean={g_x['mean']:.4f}  median={g_x['median']:.4f}")
    print(
        f"||xg - x||          mean={g_diff['mean']:.4f}  median={g_diff['median']:.4f}  "
        f"p10={g_diff['p10']:.4f}  p90={g_diff['p90']:.4f}"
    )
    print(
        f"rel=||xg-x||/||x||  mean={g_rel['mean']:.6f} median={g_rel['median']:.6f} "
        f"p10={g_rel['p10']:.6f} p90={g_rel['p90']:.6f}"
    )
    print(f"RMS(xg - x)         mean={g_rms['mean']:.6f} median={g_rms['median']:.6f}")
    print(f"cos(x, xg)          mean={g_cos['mean']:.6f} median={g_cos['median']:.6f}")
    print("=========================================================\n")

    # Sort per-prompt rows by relative L2 mean (descending)
    prompt_rows.sort(
        key=lambda r: r["rel_l2_mean"] if math.isfinite(float(r["rel_l2_mean"])) else -1.0,
        reverse=True,
    )

    # Write per-prompt CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "prompt",
                "count",
                "diff_l2_mean",
                "diff_l2_median",
                "diff_l2_p10",
                "diff_l2_p90",
                "x_l2_mean",
                "rel_l2_mean",
                "rel_l2_median",
                "rel_l2_p10",
                "rel_l2_p90",
                "rms_mean",
                "rms_median",
                "cos_mean",
                "cos_median",
            ]
        )
        for r in prompt_rows:
            w.writerow(
                [
                    r["prompt"],
                    r["count"],
                    f"{r['diff_l2_mean']:.6f}",
                    f"{r['diff_l2_median']:.6f}",
                    f"{r['diff_l2_p10']:.6f}",
                    f"{r['diff_l2_p90']:.6f}",
                    f"{r['x_l2_mean']:.6f}",
                    f"{r['rel_l2_mean']:.8f}",
                    f"{r['rel_l2_median']:.8f}",
                    f"{r['rel_l2_p10']:.8f}",
                    f"{r['rel_l2_p90']:.8f}",
                    f"{r['rms_mean']:.8f}",
                    f"{r['rms_median']:.8f}",
                    f"{r['cos_mean']:.8f}",
                    f"{r['cos_median']:.8f}",
                ]
            )

    print(f"Wrote per-prompt magnitude CSV to: {args.out_csv}")

    # Optional per-item CSV
    if args.out_items_csv:
        with open(args.out_items_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "seed_folder", "prompt", "diff_l2", "x_l2", "rel_l2", "rms", "cos_x_xg"])
            for r in per_item_rows:
                w.writerow(
                    [
                        r["file"],
                        r["seed_folder"],
                        r["prompt"],
                        f"{float(r['diff_l2']):.8f}",
                        f"{float(r['x_l2']):.8f}",
                        f"{float(r['rel_l2']):.10f}",
                        f"{float(r['rms']):.10f}",
                        f"{float(r['cos_x_xg']):.10f}",
                    ]
                )
        print(f"Wrote per-item CSV to: {args.out_items_csv}")

    print("Tip: If rel_l2 is extremely tiny (e.g., 1e-5), then golden ~= normal.")
    print("     If rel_l2 is noticeably larger (e.g., 1e-3 to 1e-1), then golden differs meaningfully from normal.")


if __name__ == "__main__":
    main()
