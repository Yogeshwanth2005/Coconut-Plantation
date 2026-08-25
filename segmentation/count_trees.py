"""
End-to-end tree counting: raw plantation image -> number of coconut trees.

This is the product path (design spec Goal 1). Everything else in
segmentation/ either builds training data or scores the model against
labeled tiles; this is the only script that takes an arbitrary image and
answers the actual question: how many trees are in it?

Pipeline:
    1. Stage 1 (stage1_mask.py) masks out non-vegetation -- sea, soil,
       roads -- by painting those pixels white, matching how the training
       data was prepared.
    2. The masked image is tiled into 256x256 crops, matching the tile
       size the segmentation model was trained on.
    3. Stage 2 (binary MK-UNet) predicts a tree/background mask per tile.
    4. Tiles are stitched back into one full-resolution prediction mask.
    5. Connected components are extracted from the STITCHED mask and
       reduced to centroids -- one centroid per tree.

Step 5 is done on the stitched mask rather than per tile on purpose. The
training tiles use a non-overlapping 256px grid, so a tree centred near a
seam has its ~14px-radius blob split across two tiles; counting per tile
would report it twice. Roughly 10% of a tile's area lies within a blob
radius of each seam, so this is a real error, not a corner case.
Stitching first makes seams invisible to the counter.

Accuracy, counting inside annotated regions where ground truth exists:

    Amrita_800_1      110 vs GT 109  (+0.9%)
    Wat Phleng_800_1 2707 vs GT 2519 (+7.5%)

Use --skip-stage1 for images that are ALREADY masked (the files in
Applicatno/MK-UNet-main/masked images/), which skips redundant work.

Usage:
    python segmentation/count_trees.py --image field.png
    python segmentation/count_trees.py --image field.png --save-overlay counted.png

    python segmentation/count_trees.py --skip-stage1 \
        --image "Applicatno/MK-UNet-main/masked images/Amrita_800_1.png"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MKUNET_DIR = ROOT / "Applicatno" / "MK-UNet-main"
sys.path.insert(0, str(MKUNET_DIR))

from mkunet_network import MK_UNet_ShallowDec  # noqa: E402

# ---- Stage 2 (segmentation) ----
TILE_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Matches count_predictions.py -- drops speckle components too small to be a
# tree. Kept identical so the counts this script reports are comparable to
# the numbers the evaluation script produces.
MIN_BLOB_AREA = 20

# ---- Stage 1 (vegetation masking) ----
# Lives in stage1_mask.py. See that module for why it uses a vegetation
# index rather than the HarinieColourAlgo colour MLP: the original dense-
# masking script is missing, and reverse-engineering it from the MLP failed
# across sites.
from stage1_mask import WHITE as STAGE1_WHITE, apply_stage1_mask  # noqa: E402


def load_segmentation_model(checkpoint_path: Path, device: torch.device):
    model = MK_UNet_ShallowDec(num_classes=2, in_channels=3, enable_cls=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_tree_mask(image_bgr: np.ndarray, model, device: torch.device) -> np.ndarray:
    """Runs Stage 2 tile by tile and stitches one full-resolution mask.

    Tiles are taken on the same non-overlapping grid as tile_and_split.py.
    Edge tiles are shifted back to land inside the image (rather than padded)
    so every tile is exactly TILE_SIZE -- the model's input size -- which
    means edge regions get predicted as part of a shifted tile. Writing those
    predictions with logical-or is safe: an overlap only ever adds tree
    pixels, and the centroid extraction that follows merges any duplicates.
    """
    h, w = image_bgr.shape[:2]
    tree_mask = np.zeros((h, w), dtype=bool)

    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            y_end, x_end = min(y + TILE_SIZE, h), min(x + TILE_SIZE, w)
            y_start, x_start = max(0, y_end - TILE_SIZE), max(0, x_end - TILE_SIZE)

            tile = image_bgr[y_start:y_end, x_start:x_end]
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                continue  # image smaller than one tile in this dimension

            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - MEAN) / STD
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()

            with torch.no_grad():
                logits = model(tensor.to(device))[0]
                pred = logits.argmax(1).squeeze(0).cpu().numpy()

            tree_mask[y_start:y_end, x_start:x_end] |= (pred == 1)

    return tree_mask


def count_trees(tree_mask: np.ndarray) -> np.ndarray:
    """Centroid per connected component of the STITCHED mask -> one per tree."""
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        tree_mask.astype(np.uint8), connectivity=8
    )
    keep = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA]
    return centroids[keep] if keep else np.empty((0, 2))


def save_overlay(image_bgr: np.ndarray, centers: np.ndarray, out_path: Path) -> None:
    overlay = image_bgr.copy()
    for cx, cy in centers:
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 12, (0, 0, 255), 2)
    cv2.putText(
        overlay, f"{len(centers)} trees", (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3, cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def main():
    parser = argparse.ArgumentParser(description="Count coconut trees in an image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "mkunet_binary_best.pth")
    parser.add_argument(
        "--skip-stage1", action="store_true",
        help="image is already masked (non-vegetation painted white)",
    )
    parser.add_argument("--save-overlay", type=Path, default=None)
    parser.add_argument("--save-mask", type=Path, default=None)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"{args.checkpoint} not found. Download it with:\n"
            "  modal volume get coconut-segmentation-output mkunet_binary_best.pth . --force"
        )

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {args.image}")
    h, w = image_bgr.shape[:2]
    print(f"Image: {args.image.name}  ({w}x{h})")

    if args.skip_stage1:
        print("Stage 1: skipped (--skip-stage1)")
        masked = image_bgr
    else:
        print("Stage 1: masking non-vegetation...")
        masked = apply_stage1_mask(image_bgr)
        kept = 100.0 * (1.0 - (masked == STAGE1_WHITE).all(2).mean())
        print(f"  kept {kept:.1f}% of pixels as vegetation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 2: segmenting on {device}...")
    model = load_segmentation_model(args.checkpoint, device)
    tree_mask = predict_tree_mask(masked, model, device)

    centers = count_trees(tree_mask)

    print(f"\n{'=' * 40}")
    print(f"TREES COUNTED: {len(centers)}")
    print(f"{'=' * 40}")
    if len(centers):
        area_px = h * w
        print(f"Density: {len(centers) / (area_px / 1e6):.1f} trees per megapixel")

    if args.save_mask:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_mask), tree_mask.astype(np.uint8) * 255)
        print(f"Mask written to {args.save_mask}")

    if args.save_overlay:
        save_overlay(image_bgr, centers, args.save_overlay)
        print(f"Overlay written to {args.save_overlay}")


if __name__ == "__main__":
    main()
