# analyze_golden_seeds.py
# Direction-stability analysis across multiple seed folders.
#
# For each prompt, we look at displacement vectors:
#   x  := x_T
#   xg := x_T_target
#   d  := xg - x
#
# Metrics per prompt:
#   - DirStab: mean pairwise cosine similarity of unit(d) across seeds
#   - CV_norm: coefficient of variation of ||d|| across seeds
#   - EVR1: explained variance ratio of the first PCA component of d across seeds
#   - Retrieval acc (optional): nearest-prototype direction classification across prompts

import os
import argparse
from collections import defaultdict

import torch


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # a,b: (..., D)
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


def pairwise_dir_stability(unit_vecs: torch.Tensor) -> float:
    """
    unit_vecs: (S, D) already normalized (or close)
    returns mean cosine over all pairs s<s'
    """
    S = unit_vecs.shape[0]
    if S < 2:
        return float("nan")
    # full cosine matrix (S,S) = unit_vecs @ unit_vecs.T
    C = unit_vecs @ unit_vecs.t()
    idx = torch.triu_indices(S, S, offset=1)
    vals = C[idx[0], idx[1]]
    return vals.mean().item()


def cv_of_norm(norms: torch.Tensor, eps: float = 1e-12) -> float:
    # norms: (S,)
    m = norms.mean().item()
    s = norms.std(unbiased=False).item()
    return s / (m + eps)


def pca_evr1(vecs: torch.Tensor, eps: float = 1e-12) -> float:
    """
    vecs: (S, D) (NOT necessarily centered)
    returns explained variance ratio of the first principal component.
    """
    X = vecs - vecs.mean(dim=0, keepdim=True)
    S = X.shape[0]
    if S < 2:
        return float("nan")
    # covariance in sample space: (S,S)
    C = (X @ X.t()) / (S - 1 + eps)
    evals = torch.linalg.eigvalsh(C).clamp_min(0)
    total = evals.sum().item()
    if total <= 0:
        return 0.0
    return evals.max().item() / total


def load_pairs(folder: str, x_key: str = "x_T", target_key: str = "x_T_target", max_prompts: int = 0):
    """
    Returns dict: prompt -> list of (file_name, d_vec_flat, norm)
    where d = x_T_target - x_T
    """
    data = defaultdict(list)
    pt_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    pt_files.sort()

    if max_prompts and max_prompts > 0:
        pt_files = pt_files[:max_prompts]

    for f in pt_files:
        path = os.path.join(folder, f)
        obj = torch.load(path, map_location="cpu", weights_only=False)

        if x_key not in obj or target_key not in obj:
            raise KeyError(
                f"Missing keys in {path}. Need '{x_key}' and '{target_key}'. "
                f"Got keys={list(obj.keys())[:40]}"
            )

        x = obj[x_key].float()
        xg = obj[target_key].float()
        prompt = obj.get("prompt", "")

        d = (xg - x).reshape(-1)  # (D,)
        n = d.norm()
        data[prompt].append((f, d, n))

    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Parent directory containing golden_pairs* folders")
    ap.add_argument(
        "--folders",
        type=str,
        nargs="+",
        required=True,
        help="Folder names under --root, e.g., golden_pairs0 ... golden_pairs4",
    )
    ap.add_argument("--x_key", type=str, default="x_T")
    ap.add_argument("--target_key", type=str, default="x_T_target")
    ap.add_argument("--max_prompts", type=int, default=0, help="Limit number of .pt per folder (0 = all)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda (analysis is usually fine on cpu)")
    ap.add_argument("--out", type=str, default="", help="Optional: path to save a TSV")
    args = ap.parse_args()

    folders = [os.path.join(args.root, f) for f in args.folders]
    for fd in folders:
        if not os.path.isdir(fd):
            raise FileNotFoundError(f"Missing folder: {fd}")

    # Load each seed-folder
    per_seed = []
    for fd in folders:
        print(f"[load] {fd}")
        per_seed.append(
            load_pairs(fd, x_key=args.x_key, target_key=args.target_key, max_prompts=args.max_prompts)
        )

    # Find prompts that exist in ALL folders
    prompt_sets = [set(d.keys()) for d in per_seed]
    common_prompts = sorted(list(set.intersection(*prompt_sets)))
    print(f"\nCommon prompts across {len(folders)} folders: {len(common_prompts)}")

    # For each prompt, we need exactly one sample per seed.
    # If each folder has multiple .pt for the same prompt, we take the first one.
    S = len(folders)

    rows = []
    proto_vecs = {}  # prompt -> prototype unit direction (D,)
    per_prompt_stats = {}

    for p in common_prompts:
        d_list = []
        n_list = []
        for s in range(S):
            entries = per_seed[s][p]
            if len(entries) == 0:
                break
            _, d, n = entries[0]
            d_list.append(d)
            n_list.append(n)

        if len(d_list) != S:
            continue

        Dmat = torch.stack(d_list, dim=0).to(args.device)       # (S, D)
        norms = torch.stack(n_list, dim=0).to(args.device)      # (S,)
        unit = Dmat / (norms.unsqueeze(-1) + 1e-12)

        dirstab = pairwise_dir_stability(unit)
        cvn = cv_of_norm(norms)
        evr1 = pca_evr1(Dmat)

        # prototype direction for retrieval: mean of unit vectors then renormalize
        proto = unit.mean(dim=0)
        proto = proto / (proto.norm() + 1e-12)
        proto_vecs[p] = proto.detach().cpu()

        per_prompt_stats[p] = {
            "DirStab": dirstab,
            "CV_norm": cvn,
            "Mean_norm": norms.mean().item(),
            "EVR1": evr1,
        }
        rows.append((p, dirstab, cvn, norms.mean().item(), evr1))

    print(f"Usable prompts (after alignment): {len(rows)}")

    # Prompt retrieval: for each (prompt p, seed s) direction, find nearest prototype among all prompts.
    prompts_used = [r[0] for r in rows]
    if len(prompts_used) >= 2:
        P = len(prompts_used)
        proto_mat = torch.stack([proto_vecs[p] for p in prompts_used], dim=0).to(args.device)  # (P,D)

        correct = 0
        total = 0
        for i, p in enumerate(prompts_used):
            for s in range(S):
                _, d, n = per_seed[s][p][0]
                v = (d.to(args.device) / (n.to(args.device) + 1e-12)).unsqueeze(0)  # (1,D)
                sims = cosine_sim(v, proto_mat).squeeze(0)  # (P,)
                pred = int(torch.argmax(sims).item())
                if pred == i:
                    correct += 1
                total += 1

        acc = correct / max(total, 1)
        chance = 1.0 / P
    else:
        acc = float("nan")
        chance = float("nan")

    # Aggregate summaries
    if rows:
        dirstabs = torch.tensor([r[1] for r in rows], dtype=torch.float32)
        cvns = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        mean_norms = torch.tensor([r[3] for r in rows], dtype=torch.float32)
        evr1s = torch.tensor([r[4] for r in rows], dtype=torch.float32)

        print("\n==== Summary over prompts ====")
        print(f"DirStab   mean={dirstabs.mean():.4f}  median={dirstabs.median():.4f}")
        print(f"CV(norm)  mean={cvns.mean():.4f}      median={cvns.median():.4f}")
        print(f"Mean||d|| mean={mean_norms.mean():.4f} median={mean_norms.median():.4f}")
        print(f"EVR1      mean={evr1s.mean():.4f}     median={evr1s.median():.4f}")
        print(f"\nPrompt retrieval Top-1 acc over seeds: {acc:.4f} (chance ~ {chance:.6f})")

    # Optional save (TSV)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("prompt\tDirStab\tCV_norm\tMean_norm\tEVR1\n")
            for p, dirstab, cvn, mn, evr1 in rows:
                f.write(f"{p}\t{dirstab:.6f}\t{cvn:.6f}\t{mn:.6f}\t{evr1:.6f}\n")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
