# train.py
"""
NPNet training script (Proposal version)

- Mix supervision: golden : weak = 2 : 8 (controlled by WEAK_SUPERVISION_RATIO)
- Cosine LR schedule + warmup
- Tau cosine decay (for temporal regularization weight)
- Loss: Charbonnier + Temporal DCT low-frequency gradient regularizer
- All float32

Assumes:
- config.py provides all hyperparameters and dirs
- data.py provides PrecomputedNoiseDataset, safe_collate
- model.py provides NPNetV
- loss.py provides NPNetVLoss
"""

import os
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from model import NPNetV
from data import PrecomputedNoiseDataset, safe_collate
from loss import NPNetVLoss

torch.set_default_dtype(torch.float32)


# -----------------------------
# schedules
# -----------------------------
def cosine_with_warmup(step: int, total: int, base: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base * float(step + 1) / float(max(1, warmup))
    prog = (step - warmup) / float(max(1, total - warmup))
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * prog))


def cosine_tau(step: int, total: int, t0: float, t1: float) -> float:
    prog = step / float(max(1, total))
    return t1 + 0.5 * (t0 - t1) * (1.0 + math.cos(math.pi * prog))


def _get_clip_path() -> str:
    """
    Priority:
      1) config.CLIP_PATH (if exists)
      2) env NPCLIP_PATH
      3) fallback default (AutoDL common path)
    """
    if hasattr(config, "CLIP_PATH") and isinstance(config.CLIP_PATH, str) and len(config.CLIP_PATH) > 0:
        return config.CLIP_PATH
    if "NPCLIP_PATH" in os.environ and os.environ["NPCLIP_PATH"].strip():
        return os.environ["NPCLIP_PATH"].strip()
    return "/root/autodl-tmp/VideoCrafter/local_clip_model"


# -----------------------------
# main
# -----------------------------
def main():
    device = config.DEVICE

    # -----------------------------
    # model
    # -----------------------------
    npnet = NPNetV(
        channels=config.CHANNELS,
        T=config.TEMPORAL_DIM,
        H=config.HEIGHT,
        W=config.WIDTH,
        freq_decay=config.FREQ_DECAY,
    ).to(device)

    criterion = NPNetVLoss(
        tau=float(config.TAU_START),
        temporal_low_freq_k=int(config.LOSS_TEMP_LOW_FREQ_K),
        charbonnier_eps=1e-6,
    ).to(device)

    optimizer = optim.AdamW(npnet.parameters(), lr=float(config.LR), weight_decay=0.01)

    # -----------------------------
    # text encoder (CLIP)
    # -----------------------------
    clip_path = _get_clip_path()
    assert os.path.exists(clip_path), f"CLIP path not found: {clip_path}"

    tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    text_encoder = CLIPTextModel.from_pretrained(clip_path, torch_dtype=torch.float32).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    # -----------------------------
    # data
    # -----------------------------
    weak_dataset = PrecomputedNoiseDataset(config.WEAK_PAIRS_DIR)
    golden_dataset = PrecomputedNoiseDataset(config.GOLDEN_PAIRS_DIR)

    weak_loader = DataLoader(
        weak_dataset,
        batch_size=int(config.BATCH_SIZE),
        shuffle=True,
        num_workers=int(config.NUM_WORKERS),
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate,
    )
    golden_loader = DataLoader(
        golden_dataset,
        batch_size=int(config.BATCH_SIZE),
        shuffle=True,
        num_workers=int(config.NUM_WORKERS),
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate,
    )

    steps_per_epoch = max(len(weak_loader), len(golden_loader))
    total_steps = steps_per_epoch * int(config.NUM_EPOCHS)
    warmup = max(100, int(0.05 * total_steps))
    min_lr = float(config.LR) * 0.1

    npnet.train()
    global_step = 0

    losses, main_losses, temp_losses, all_steps = [], [], [], []

    print("------ Training NPNet (Proposal) ------")
    print(f"device={device} | clip_path={clip_path}")
    print(f"weak_dir={config.WEAK_PAIRS_DIR}")
    print(f"golden_dir={config.GOLDEN_PAIRS_DIR}")
    print(f"epochs={config.NUM_EPOCHS} | steps/epoch={steps_per_epoch} | total_steps={total_steps}")
    print(f"mix: weak_ratio={config.WEAK_SUPERVISION_RATIO} (so golden_ratio={1.0 - config.WEAK_SUPERVISION_RATIO})")

    for epoch in range(int(config.NUM_EPOCHS)):
        print(f"\n===== Epoch {epoch + 1}/{config.NUM_EPOCHS} =====")

        weak_iter = iter(weak_loader)
        gold_iter = iter(golden_loader)

        for _ in tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}", leave=True):
            # sample supervision source: 80% weak, 20% golden by default
            if torch.rand(1).item() < float(config.WEAK_SUPERVISION_RATIO):
                try:
                    batch = next(weak_iter)
                except StopIteration:
                    weak_iter = iter(weak_loader)
                    batch = next(weak_iter)
            else:
                try:
                    batch = next(gold_iter)
                except StopIteration:
                    gold_iter = iter(golden_loader)
                    batch = next(gold_iter)

            if batch is None or batch[0] is None:
                continue

            x_T, x_T_target, texts = batch
            x_T = x_T.to(device=device, dtype=torch.float32)
            x_T_target = x_T_target.to(device=device, dtype=torch.float32)

            # text embedding (mean pooling with attention mask)
            with torch.no_grad():
                inputs = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                out = text_encoder(input_ids=inputs["input_ids"])
                hidden = out.last_hidden_state.float()  # (B,L,D)
                mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B,L,1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                E_txt = pooled.float()  # (B,D)

            # lr & tau schedules
            lr_now = cosine_with_warmup(global_step, total_steps, float(config.LR), min_lr, warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            tau_now = cosine_tau(global_step, total_steps, float(config.TAU_START), float(config.TAU_END))

            # forward + loss
            x_star = npnet(x_T, E_txt)
            loss, L_main, L_temp = criterion(x_star, x_T_target, tau_override=tau_now)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(npnet.parameters(), 1.0)
            optimizer.step()

            if global_step % int(config.LOG_STEP) == 0:
                losses.append(float(loss.item()))
                main_losses.append(float(L_main.item()))
                temp_losses.append(float(L_temp.item()))
                all_steps.append(int(global_step))

            global_step += 1

    # -----------------------------
    # save outputs
    # -----------------------------
    # loss curve
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(all_steps, losses)
    plt.title("Total Loss")

    plt.subplot(1, 3, 2)
    plt.plot(all_steps, main_losses)
    plt.title("Main Loss")

    plt.subplot(1, 3, 3)
    plt.plot(all_steps, temp_losses)
    plt.title("Temp Loss")

    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    print("Saved loss_curve.png")

    # checkpoint
    ckpt_name = getattr(config, "CKPT_NAME", "npnet_final.pth")
    torch.save(npnet.state_dict(), ckpt_name)
    print(f"Saved {ckpt_name}")


if __name__ == "__main__":
    main()
