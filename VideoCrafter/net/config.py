# config.py
"""
NPNet configuration (GitHub-friendly).

Design:
- No hard-coded machine-specific absolute paths by default.
- Paths can be overridden by environment variables.
- Training hyperparams remain explicit here for reproducibility.
"""

from __future__ import annotations

import os
import torch

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Training hyperparams
# -----------------------------
LR = 2e-4                 # AdamW base lr
BATCH_SIZE = 32
NUM_WORKERS = 4
LOG_STEP = 50
NUM_EPOCHS = 3

# Dynamic temporal regularization
TAU_START = 0.10
TAU_END = 0.02
LOSS_TEMP_LOW_FREQ_K = 4

# Weak vs golden sampling ratio (probability of sampling WEAK)
WEAK_SUPERVISION_RATIO = 0.8


# -----------------------------
# Data paths (override via env)
# -----------------------------
# Example:
#   export NPNET_WEAK_PAIRS_DIR=/path/to/weak_pairs
#   export NPNET_GOLDEN_PAIRS_DIR=/path/to/golden_pairs
WEAK_PAIRS_DIR = os.environ.get("NPNET_WEAK_PAIRS_DIR", "./data/weak_pairs")
GOLDEN_PAIRS_DIR = os.environ.get("NPNET_GOLDEN_PAIRS_DIR", "./data/golden_pairs")

# CLIP local folder (tokenizer + text encoder), override via env:
#   export NPNET_CLIP_PATH=/path/to/local_clip_model
CLIP_PATH = os.environ.get("NPNET_CLIP_PATH", "./assets/local_clip_model")


# -----------------------------
# Noise / latent shape (reference only)
# -----------------------------
# (Only for init/logging. Model can handle other sizes if written that way.)
CHANNELS = 4
TEMPORAL_DIM = 16
HEIGHT = 32
WIDTH = 32


# -----------------------------
# Frequency branch
# -----------------------------
FREQ_DECAY = 1.2


# -----------------------------
# Text encoder
# -----------------------------
# Must match the CLIP actually used during training.
TEXT_ENCODER_MODEL_DIM = 512  # e.g., CLIP-ViT-B/32 hidden size
