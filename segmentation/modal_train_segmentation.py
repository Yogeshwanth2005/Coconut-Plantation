"""
Trains a binary (tree vs. background) MK-UNet segmentation model on
Modal, reading tiled data from the 'coconut-segmentation-data' Volume
(see tile_and_split.py + generate_blob_masks.py) and writing the best
checkpoint back to the 'coconut-segmentation-output' Volume.

Replaces Applicatno/MK-UNet-main/train_plantation.py's approach:
  - Binary output (tree/background), not the old 15/16/32-class scheme
    that mostly encoded per-site color fingerprints rather than genuine
    density/arrangement patterns (see design spec for the cross-site
    holdout evidence).
  - Trains on Stage-1 color-masked images (non-green/sea/other already
    blanked) + point-derived blob masks (real tree centers), not
    box-fill masks.
  - Data comes from a site-aware split (Amrita held out entirely) built
    by tile_and_split.py -- not train_plantation.py's random shuffle of
    pooled tiles across all sites.
  - num_classes is unambiguous (binary) -- avoids the old
    train/test num_classes mismatch bug (16 vs 32).

Reuses the existing MK-UNet network code (mkunet_network.py) and
dataloader (utils/dataloader_polyp.py) via a Modal mount -- the
architecture itself was never the problem, only the data/labels/split.

See docs/superpowers/specs/2026-08-23-tree-count-segmentation-design.md.

Run:
    modal run "segmentation/modal_train_segmentation.py"

Then download the result:
    modal volume get coconut-segmentation-output mkunet_binary_best.pth . --force
"""

from pathlib import Path

import modal

app = modal.App("coconut-segmentation-train")

MKUNET_LOCAL_DIR = Path(__file__).resolve().parent.parent / "Applicatno" / "MK-UNet-main"
EMPTY_INIT = Path(__file__).resolve().parent / "_empty_init.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # required by opencv-python
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "opencv-python-headless==4.10.0.84",
        "albumentations==1.4.18",
        "timm==0.9.16",
        "numpy<2",
        "pillow",
    )
    .add_local_file(str(MKUNET_LOCAL_DIR / "mkunet_network.py"), "/root/mkunet_network.py")
    .add_local_file(
        str(MKUNET_LOCAL_DIR / "utils" / "dataloader_polyp.py"),
        "/root/utils/dataloader_polyp.py",
    )
    .add_local_file(str(EMPTY_INIT), "/root/utils/__init__.py")
)

data_volume = modal.Volume.from_name("coconut-segmentation-data", create_if_missing=True)
output_volume = modal.Volume.from_name("coconut-segmentation-output", create_if_missing=True)

VOLUME_MOUNT = "/data"
DATA_ROOT = "/data/data/tiled"  # matches `modal volume put coconut-segmentation-data <local> /data/tiled`
OUTPUT_ROOT = "/output"

RESUME_FILENAME = "mkunet_resume_state.pth"

NUM_CLASSES = 1  # binary: dataloader treats num_classes<=1 as tree-vs-background
IMG_SIZE = 256
BATCH_SIZE = 8
MAX_EPOCHS = 60
PATIENCE = 12
LEARNING_RATE = 5e-4
SEED = 42


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: data_volume, OUTPUT_ROOT: output_volume},
    timeout=60 * 60 * 3,
)
def train():
    import sys

    sys.path.insert(0, "/root")

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from mkunet_network import MK_UNet_ShallowDec
    from utils.dataloader_polyp import get_loader

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RESUME_PATH = Path(OUTPUT_ROOT) / RESUME_FILENAME

    data_root = Path(DATA_ROOT)
    train_loader = get_loader(
        image_root=str(data_root / "train" / "images"),
        gt_root=str(data_root / "train" / "masks"),
        batchsize=BATCH_SIZE,
        trainsize=IMG_SIZE,
        shuffle=True,
        num_workers=2,
        augmentation=True,
        split="train",
        num_classes=NUM_CLASSES,
        preload=False,
    )
    val_loader = get_loader(
        image_root=str(data_root / "val" / "images"),
        gt_root=str(data_root / "val" / "masks"),
        batchsize=BATCH_SIZE,
        trainsize=IMG_SIZE,
        shuffle=False,
        num_workers=2,
        augmentation=False,
        split="val",
        num_classes=NUM_CLASSES,
        preload=False,
    )
    print(f"Train tiles: {len(train_loader.dataset)}  |  Val tiles: {len(val_loader.dataset)}")

    # ---- Foreground/background class weight (tree coverage is only ~2.3%) ----
    # The raw inverse-frequency weight ((1-0.023)/0.023 ~= 42.5) combined with
    # dice loss double-counts the imbalance correction: visual review of
    # predictions at IoU~0.16 showed large, merged, oversized blobs and
    # false alarms in tree-sparse tiles -- the model over-predicting "tree"
    # broadly rather than drawing tight per-tree blobs. Dice loss already
    # handles class imbalance on its own, so cross-entropy doesn't need the
    # full inverse-frequency weight on top of it. Using sqrt of the raw
    # weight as a gentler compromise.
    fg_weight = ((1 - 0.023) / 0.023) ** 0.5  # ~6.5
    class_weights = torch.tensor([1.0, fg_weight], dtype=torch.float32).to(device)
    print(f"Class weights [background, tree]: {class_weights.cpu().numpy().round(2)}")

    # ---- Model: binary segmentation, no classification head ----
    model = MK_UNet_ShallowDec(num_classes=2, in_channels=3, enable_cls=False)
    model = model.to(device)

    def dice_loss(pred_softmax, target, num_classes=2):
        loss = 0.0
        valid = 0
        for c in range(num_classes):
            pred_c = pred_softmax[:, c]
            mask_c = (target == c).float()
            if mask_c.sum() == 0 and pred_c.sum() < 1e-6:
                continue
            intersection = (pred_c * mask_c).sum()
            union = pred_c.sum() + mask_c.sum()
            loss += 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
            valid += 1
        return loss / max(valid, 1)

    def combined_loss(logits, mask):
        target = mask.squeeze(1).long()
        ce = F.cross_entropy(logits, target, weight=class_weights)
        dice = dice_loss(F.softmax(logits, dim=1), target)
        return ce + dice

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    def tree_iou(logits, mask):
        pred = logits.argmax(1)
        target = mask.squeeze(1).long()
        pred_fg = (pred == 1)
        target_fg = (target == 1)
        intersection = (pred_fg & target_fg).sum().item()
        union = (pred_fg | target_fg).sum().item()
        return intersection / union if union > 0 else float("nan")

    best_val_iou = -1.0
    epochs_without_improvement = 0
    best_state = None
    start_epoch = 0

    # ---- Resume from a prior preemption, if a resume checkpoint exists ----
    # T4 spot containers get preempted mid-run (observed twice already); without
    # this, a preemption silently restarts from epoch 0 and throws away
    # everything trained so far. Saved every epoch, not just on improvement, so
    # a resume picks up right where training left off regardless of whether the
    # most recent epoch was a new best.
    if RESUME_PATH.exists():
        print(f"Found resume checkpoint at {RESUME_PATH}, resuming...")
        resume_state = torch.load(RESUME_PATH, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        start_epoch = resume_state["epoch"] + 1
        best_val_iou = resume_state["best_val_iou"]
        epochs_without_improvement = resume_state["epochs_without_improvement"]
        best_state = resume_state["best_model_state"]
        print(f"Resumed at epoch {start_epoch + 1}/{MAX_EPOCHS}, best_val_iou so far: {best_val_iou:.4f}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)[0]
            loss = combined_loss(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / n_batches
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_ious = []
        n_val_batches = 0
        with torch.no_grad():
            for images, masks, _shape, _name in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)[0]
                val_loss_sum += combined_loss(outputs, masks).item()
                n_val_batches += 1
                iou = tree_iou(outputs, masks)
                if not np.isnan(iou):
                    val_ious.append(iou)

        val_loss = val_loss_sum / n_val_batches
        val_iou = float(np.mean(val_ious)) if val_ious else 0.0

        print(
            f"Epoch {epoch + 1:3d}/{MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_tree_iou={val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save resume state every epoch (not just on improvement) so a
        # preemption never loses more than one epoch of progress.
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_iou": best_val_iou,
                "epochs_without_improvement": epochs_without_improvement,
                "best_model_state": best_state,
            },
            RESUME_PATH,
        )
        output_volume.commit()

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
            break

    print(f"\nBest val tree IoU: {best_val_iou:.4f}")

    out_path = Path(OUTPUT_ROOT) / "mkunet_binary_best.pth"
    torch.save(best_state, out_path)
    RESUME_PATH.unlink(missing_ok=True)  # training finished normally, resume state no longer needed
    output_volume.commit()
    print(f"Saved best checkpoint to {out_path}")

    return {"best_val_iou": best_val_iou}


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"\nTraining finished. Best val tree IoU: {result['best_val_iou']:.4f}")
    print("Download with:")
    print("  modal volume get coconut-segmentation-output mkunet_binary_best.pth . --force")
