"""
Compares predicted vs. ground-truth tree COUNTS per val tile, instead of
pixel IoU. inspect_predictions.py showed the model often gets blob
locations roughly right (or roughly wrong) even when pixel overlap is
poor -- this asks a more forgiving, count-relevant question: if you were
counting trees from these predictions, how close would you be?

Each connected component in a mask (pred or GT) is reduced to its
centroid. Predicted centroids are matched to GT centroids via optimal
(Hungarian) assignment, restricted to pairs within MATCH_DIST_PX of each
other. Unmatched GT blobs are misses (FN), unmatched pred blobs are
false alarms (FP), matched pairs are hits (TP) -- giving per-tile and
aggregate count precision/recall/F1, plus raw predicted-vs-actual count
totals.

MATCH_DIST_PX=25 (~half the median 49.7px nearest-neighbor tree spacing
from generate_blob_masks.py) -- close enough to plausibly be the same
tree, tight enough not to match distinct neighboring trees.

Only ANNOTATED regions are scored. Masks carry a 128 "ignore" band marking
pixels no annotator ever looked at (see generate_blob_masks.py); predicted
trees centred there are dropped instead of counted as false alarms, because
those regions frequently do contain real, unlabeled trees. Metrics from
before this change over-counted FPs and are not comparable.

Usage:
    python segmentation/count_predictions.py --checkpoint mkunet_binary_best.pth
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
MKUNET_DIR = ROOT / "Applicatno" / "MK-UNet-main"
sys.path.insert(0, str(MKUNET_DIR))

from mkunet_network import MK_UNet_ShallowDec  # noqa: E402

TILED_ROOT = Path(__file__).resolve().parent / "tiled"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MATCH_DIST_PX = 25
MIN_BLOB_AREA = 20  # drop tiny speckle components (noise, not a tree)

# Masks mark never-annotated pixels with this value (see
# generate_blob_masks.py). Predictions whose centroid falls in an ignore
# region are discarded rather than counted as false alarms: nobody labeled
# those pixels, and inspection showed such regions are frequently dense,
# genuinely unlabeled coconut canopy. Counting them as FPs is what made the
# earlier precision figures uninterpretable.
IGNORE_VALUE = 128


def load_model(checkpoint_path: Path, device: torch.device):
    model = MK_UNet_ShallowDec(num_classes=2, in_channels=3, enable_cls=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def blob_centroids(mask: np.ndarray) -> np.ndarray:
    """Connected-component centroids for a binary (0/255 or 0/1) mask, filtering tiny specks."""
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    keep = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA]
    return centroids[keep] if keep else np.empty((0, 2))


def drop_points_in_ignore(points: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """Removes predicted centroids landing on never-annotated pixels."""
    if len(points) == 0:
        return points
    h, w = ignore.shape
    xs = np.clip(points[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(points[:, 1].round().astype(int), 0, h - 1)
    return points[~ignore[ys, xs]]


def match_counts(pred_pts: np.ndarray, gt_pts: np.ndarray, max_dist: float) -> tuple[int, int, int]:
    """Returns (tp, fp, fn) via Hungarian matching restricted to max_dist."""
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0

    dists = np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=2)
    cost = np.where(dists <= max_dist, dists, max_dist * 10)
    row_ind, col_ind = linear_sum_assignment(cost)

    tp = sum(1 for r, c in zip(row_ind, col_ind) if dists[r, c] <= max_dist)
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("mkunet_binary_best.pth"))
    parser.add_argument("--site", type=str, default=None, help="only evaluate tiles whose filename starts with this site prefix")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="which tiled split to evaluate")
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

    total_tp = total_fp = total_fn = 0
    total_pred_count = total_gt_count = 0
    per_tile_errors = []

    for img_path in image_files:
        mask_path = split_masks / img_path.name
        image_bgr = cv2.imread(str(img_path))
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        tensor = preprocess(image_bgr).to(device)
        with torch.no_grad():
            logits = model(tensor)[0]
            pred = logits.argmax(1).squeeze(0).cpu().numpy()
        pred_mask = (pred == 1).astype(np.uint8) * 255

        # GT trees are the 255 blobs; the 128 band is unannotated, not empty.
        ignore = gt_mask == IGNORE_VALUE
        gt_pts = blob_centroids((gt_mask == 255).astype(np.uint8) * 255)
        pred_pts = blob_centroids(pred_mask)
        pred_pts = drop_points_in_ignore(pred_pts, ignore)

        if len(gt_pts) == 0 and len(pred_pts) == 0:
            continue  # fully-unannotated tile contributes nothing either way

        tp, fp, fn = match_counts(pred_pts, gt_pts, MATCH_DIST_PX)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_pred_count += len(pred_pts)
        total_gt_count += len(gt_pts)

        per_tile_errors.append(len(pred_pts) - len(gt_pts))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    mean_abs_err = np.mean(np.abs(per_tile_errors))
    mean_err = np.mean(per_tile_errors)  # signed: positive = overcounting

    print(
        f"Evaluated {len(image_files)} {args.split} tiles "
        f"(match distance <= {MATCH_DIST_PX}px; unannotated regions excluded)"
    )
    print(f"\nTotal GT trees:   {total_gt_count}")
    print(f"Total pred trees: {total_pred_count}  ({100 * (total_pred_count - total_gt_count) / total_gt_count:+.1f}% vs GT)")
    print(f"\nMatched (TP): {total_tp}   Missed (FN): {total_fn}   False alarms (FP): {total_fp}")
    print(f"\nCount precision: {precision:.3f}")
    print(f"Count recall:    {recall:.3f}")
    print(f"Count F1:        {f1:.3f}")
    print(f"\nPer-tile count error -- mean absolute: {mean_abs_err:.2f} trees/tile, mean signed: {mean_err:+.2f} trees/tile")


if __name__ == "__main__":
    main()
