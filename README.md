# Semantic Noise Diagnostics for Video Diffusion Models

This repository provides tools to **generate, consume, and analyze structured noise pairs**
for large-scale video diffusion models.

The goal is to study **semantic noise geometry, temporal consistency, and correction behavior**
across different diffusion backends, without modifying the underlying models.

---

## Workflow Overview

The codebase follows a simple, reproducible workflow:

1. Generate structured noise pairs from a diffusion backend
2. Re-run diffusion starting from the same noise pairs
3. Analyze noise geometry and temporal behavior
4. Train and evaluate NP-Net using noise pairs

All steps are modular and backend-agnostic.

---

## Repository Structure

```
Analysis/        Analysis scripts (run directly on saved .pt files)
OpenSora/        OpenSora pair generation and inference
VideoCrafter/    VideoCrafter pair generation, inference, and NPNet
README.md
```

---

## Dependencies

### Diffusion Backend Dependencies

- **OpenSora**: Follow the [official installation guide](https://github.com/hpcaitech/Open-Sora)
- **VideoCrafter**: Follow the [official installation guide](https://github.com/AILab-CVC/VideoCrafter)

All diffusion-related dependencies are inherited from the selected backend and should be installed following their official instructions.

### Optional Evaluation Tools

For downstream analysis (not required for core functionality):
- VBench
- CLIP-based metrics

---

## OpenSora Backend

OpenSora is used **only** for noise-pair generation and inference from pre-generated pairs.
The OpenSora codebase itself is **not modified**.

### Setup

Clone OpenSora anywhere on your machine:

```bash
git clone https://github.com/hpcaitech/Open-Sora.git
cd Open-Sora
# Follow OpenSora's installation instructions
```

Set environment variable:

```bash
export OPENSORA_ROOT=/path/to/Open-Sora
```

### Step 1: Generate golden pairs (OpenSora)

```bash
python OpenSora/make_golden_pairs.py \
  --config configs/diffusion/inference/256px.py \
  --prompt_file prompts.txt \
  --outdir pairs_golden_opensora \
  --K_steps 10 --total_steps 50 \
  --cfg_forward 7.5 --cfg_backward 1.0 \
  --height 256 --width 256 --num_frames 64
```

**Note**: `configs/diffusion/inference/256px.py` refers to OpenSora's config file located in `$OPENSORA_ROOT/configs/diffusion/inference/256px.py`.

**Output**: Generates `.pt` files in `pairs_golden_opensora/`, each containing:
- `x_T`: Initial noise tensor
- `x_T_target`: Target noise after correction
- `prompt`: Text prompt used
- `seed`: Random seed

### Step 2: Inference from pairs (OpenSora)

```bash
python OpenSora/inference.py \
  --config configs/diffusion/inference/256px.py \
  --pairs_dir pairs_golden_opensora \
  --savedir outputs_opensora_pairs \
  --num_steps 50 --guidance 4.0
```

**Output**: Generated videos in `outputs_opensora_pairs/`

---

## VideoCrafter Backend

VideoCrafter is used for pair generation, pair-based inference, and NP-Net training.

### Setup

Clone VideoCrafter:

```bash
git clone https://github.com/AILab-CVC/VideoCrafter.git
cd VideoCrafter
# Follow VideoCrafter's installation instructions
```

Set environment variable (recommended):

```bash
export VIDEOCRAFTER_ROOT=/path/to/VideoCrafter
```

Copy and paste the Work directory to VideoCrafter

### Step 3: Generate pairs (VideoCrafter)

**Golden pairs**:

```bash
python VideoCrafter/Work/make_golden_pairs.py \
  --videocrafter_root /path/to/VideoCrafter \
  --prompt_file prompts.txt \
  --outdir pairs_golden_vc
```

**Weak pairs**:

```bash
python VideoCrafter/Work/make_weak_pairs.py \
  --videocrafter_root /path/to/VideoCrafter \
  --prompt_file prompts.txt \
  --outdir pairs_weak_vc
```

**Output**: Each `.pt` file contains:
- `x_T`: Shape `[C, T, H, W]` - initial latent noise
- `x_T_target`: Shape `[C, T, H, W]` - corrected noise
- `prompt`: Text prompt
- `meta`: meta data

### Step 4: Inference from pairs (VideoCrafter)

```bash
python VideoCrafter/Work/inference.py \
  --videocrafter_root /path/to/VideoCrafter \
  --pairs_dir pairs_golden_vc \
  --savedir outputs_vc_pairs
```

**Output**: Generated videos in `outputs_vc_pairs/`

---

## NPNet (VideoCrafter/net)

NPNet learns to predict noise corrections from weak pairs to golden pairs.

### Step 5: Train NPNet

```bash
export NPNET_WEAK_PAIRS_DIR=pairs_weak_vc
export NPNET_GOLDEN_PAIRS_DIR=pairs_golden_vc

python VideoCrafter/net/train.py
```

**Note**: Paths can be absolute or relative. If relative, they are resolved from the current working directory.

**Output**:
- `npnet_final.pth`: Trained model checkpoint
- Training logs and loss curves

### Step 6: NPNet inference

```bash
python VideoCrafter/net/inference_net.py \
  --pairs_dir pairs_golden_vc \
  --ckpt npnet_final.pth
```

**Output**: Predicted noise corrections saved in the same format as input pairs

---

## Analysis

All analysis scripts operate **directly on saved `.pt` noise pairs** and do not require
any diffusion model.

### Analysis Scripts

**`golden_normal_diff.py`**  
Compare `x_T` vs `x_T_target` (magnitude + cosine similarity). Sanity-check correction scale.

```bash
python Analysis/golden_normal_diff.py --pairs_dir pairs_golden_vc
```

Output: Statistics on noise displacement magnitude and direction.

---

**`direction.py`**  
Directional stability across seeds (DirStab / EVR1-style geometry checks).

```bash
python Analysis/direction.py --pairs_dir pairs_golden_vc
```

Output: Geometric consistency metrics across different random seeds.

---

**`analyze_latent_freq_pt.py`**  
Frequency analysis of latent displacement (spatial vs temporal energy, HF ratios).

```bash
python Analysis/analyze_latent_freq_pt.py --pairs_dir pairs_golden_vc
```

Output: Frequency spectrum analysis showing spatial and temporal energy distribution.

---

**`analyze_spatiotemporal_flicker_pt.py`**  
Flicker diagnostics: quantify temporal high-frequency jitter vs spatial structure.

```bash
python Analysis/analyze_spatiotemporal_flicker_pt.py --pairs_dir pairs_golden_vc
```

Output: Temporal stability metrics and flicker quantification.

---

<!-- ## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{golden-noise-transfer,
  author = {klkds},
  title = {Semantic Noise Diagnostics for Video Diffusion Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/klkds/golden-noise-transfer}
}
```

--- -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Notes

* All steps are independent and can be run separately
* Pair files generated by OpenSora and VideoCrafter share the same interface
* Analysis scripts are fully backend-agnostic
* The repository does not modify any underlying diffusion model code
* All noise operations are performed in latent space

---

## Acknowledgments

This work builds upon:
- [OpenSora](https://github.com/hpcaitech/Open-Sora)
- [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter)

Special thanks to the maintainers of these excellent diffusion model implementations.