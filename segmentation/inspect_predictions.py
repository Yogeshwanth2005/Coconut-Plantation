"""
Loads the trained binary MK-UNet checkpoint and runs it on a handful of
val tiles, saving side-by-side (input | ground truth | prediction) panels
so you can visually judge what IoU ~0.16 actually looks like: fuzzy-but-
located blobs (fixable, maybe fine for counting) vs. genuinely missed
trees (a real problem).

Usage:
    python segmentation/inspect_predictions.py \
        --checkpoint mkunet_binary_best.pth \
        --n 12

Requires the checkpoint downloaded locally first:
    modal volume get coconut-segmentation-output mkunet_binary_best.pth . --force
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MKUNET_DIR = ROOT / "Applicatno" / "MK-UNet-main"
sys.path.insert(0, str(MKUNET_DIR))

from mkunet_network import MK_UNet_ShallowDec  # noqa: E402

TILED_ROOT = Path(__file__).resolve().parent / "tiled"
OUT_DIR = Path(__file__).resolve().parent / "prediction_review"

IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint_path: Path, device: torch.device):
    model = MK_UNet_ShallowDec(num_classes=2, in_channels=3, enable_cls=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def tree_iou(pred_fg: np.ndarray, gt_fg: np.ndarray) -> float:
    intersection = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    return intersection / union if union > 0 else float("nan")


def make_panel(image_bgr, gt_mask, pred_mask, iou_value, tile_name) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    gt_overlay = image_bgr.copy()
    gt_overlay[gt_mask > 0] = (0, 255, 0)  # green = ground truth
    gt_overlay = cv2.addWeighted(image_bgr, 0.5, gt_overlay, 0.5, 0)

    pred_overlay = image_bgr.copy()
    pred_overlay[pred_mask > 0] = (0, 0, 255)  # red = prediction
    pred_overlay = cv2.addWeighted(image_bgr, 0.5, pred_overlay, 0.5, 0)

    combined = image_bgr.copy()
    combined[np.logical_and(gt_mask > 0, pred_mask == 0)] = (0, 255, 0)   # missed (FN) = green
    combined[np.logical_and(pred_mask > 0, gt_mask == 0)] = (0, 0, 255)   # false alarm (FP) = red
    combined[np.logical_and(gt_mask > 0, pred_mask > 0)] = (0, 255, 255)  # correct (TP) = yellow

    panel = np.concatenate([image_bgr, gt_overlay, pred_overlay, combined], axis=1)
    label = f"{tile_name}  IoU={iou_value:.3f}  (yellow=hit green=missed red=false-alarm)"
    cv2.putText(panel, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("mkunet_binary_best.pth"))
    parser.add_argument("--n", type=int, default=12, help="number of random val tiles to inspect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site", type=str, default=None, help="only sample tiles whose filename starts with this site prefix")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="which tiled split to sample from")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"{args.checkpoint} not found. Download it first with:\n"
            f"  modal volume get coconut-segmentation-output mkunet_binary_best.pth . --force"
        )

    split_images = TILED_ROOT / args.split / "images"
    split_masks = TILED_ROOT / args.split / "masks"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    pattern = f"{args.site}*.png" if args.site else "*.png"
    image_files = sorted(split_images.glob(pattern))
    if not image_files:
        raise FileNotFoundError(f"No {args.split} tiles found in {split_images} matching {pattern}")

    # Bias the sample toward tiles that actually contain trees, so we're
    # not just looking at empty-background tiles (which dominate numerically).
    with_trees, without_trees = [], []
    for img_path in image_files:
        mask_path = split_masks / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        (with_trees if mask is not None and mask.max() > 0 else without_trees).append(img_path)

    random.seed(args.seed)
    n_with = min(args.n, len(with_trees))
    sample = random.sample(with_trees, n_with)
    if n_with < args.n:
        sample += random.sample(without_trees, min(args.n - n_with, len(without_trees)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ious = []

    for img_path in sample:
        mask_path = split_masks / img_path.name
        image_bgr = cv2.imread(str(img_path))
        gt_mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_fg = (gt_mask_raw >= 1) if gt_mask_raw.max() <= 127 else (gt_mask_raw > 20)

        tensor = preprocess(image_bgr).to(device)
        with torch.no_grad():
            logits = model(tensor)[0]
            pred = logits.argmax(1).squeeze(0).cpu().numpy()
        pred_fg = pred == 1

        iou = tree_iou(pred_fg, gt_fg)
        ious.append(iou)

        panel = make_panel(
            image_bgr,
            gt_fg.astype(np.uint8) * 255,
            pred_fg.astype(np.uint8) * 255,
            iou,
            img_path.stem,
        )
        cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_review.png"), panel)

    print(f"Wrote {len(sample)} review panels to {OUT_DIR}")
    valid_ious = [i for i in ious if not np.isnan(i)]
    if valid_ious:
        print(f"Mean IoU on sampled tiles: {np.mean(valid_ious):.4f}")
    print("\nLegend: green overlay = ground truth, red overlay = prediction,")
    print("in the 4th panel: yellow = correct, green = missed tree, red = false alarm")


if __name__ == "__main__":
    main()
