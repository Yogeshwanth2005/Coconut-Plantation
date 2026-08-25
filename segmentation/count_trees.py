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

Scale is handled automatically. The model learned trees ~43px apart, so an
upload at a different zoom must be resized first; find_best_scale() does
that without the user knowing any of it. Measured on a 512x512 plantation
crop with 112 known trees:

    native 512px            113  (+0.9%)
    shrunk to 256px, fixed   21  (-81%)   <- what happens without rescaling
    shrunk to 256px, auto   112  (+0.0%)
    shrunk to 192px, auto   123  (+9.8%)

Accuracy on full source images, counting inside annotated regions:

    Amrita_800_1      110 vs GT 109  (+0.9%)
    Wat Phleng_800_1 2707 vs GT 2519 (+7.5%)

LIMIT -- rescaling recovers a shrunk photo, it does not create detail that
was never captured. Below roughly 1/4 of the trained scale the crowns are
gone and no resize brings them back.

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


# The model learned trees at the source imagery's scale: measured across all
# 18 annotated tiles, neighbouring trees sit a median of 43px apart (range
# 36-58). An image at a different zoom breaks this -- trees 5px apart are
# invisible to it, trees 200px apart do not look like its training examples.
# find_best_scale() searches for the resize that puts detections near this
# spacing, so an arbitrary upload does not need the user to know any of it.
EXPECTED_TREE_SPACING_PX = 43.0

# Scales tried when auto-detecting. Spans 8x down to 1/4x, which covers a
# close-up phone photo through to a wide satellite view.
SCALE_CANDIDATES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

# A scale is only believable if it finds enough trees to measure spacing
# from; below this the "spacing" is noise from a handful of blobs.
MIN_DETECTIONS_FOR_SCALE = 8


def _median_spacing(centers: np.ndarray) -> float:
    """Median nearest-neighbour distance between detected trees."""
    if len(centers) < 2:
        return float("nan")
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(centers).query(centers, k=2)
    return float(np.median(dist[:, 1]))


def ensure_min_size(image_bgr: np.ndarray) -> np.ndarray:
    """Pads an image up to at least one tile, so small photos still run.

    Reflection padding rather than a constant colour: a white or black
    border would read as a hard edge and could invent detections along it.
    """
    h, w = image_bgr.shape[:2]
    if h >= TILE_SIZE and w >= TILE_SIZE:
        return image_bgr
    pad_y = max(0, TILE_SIZE - h)
    pad_x = max(0, TILE_SIZE - w)
    return cv2.copyMakeBorder(
        image_bgr, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT_101
    )


def find_best_scale(image_bgr: np.ndarray, model, device: torch.device) -> tuple[float, np.ndarray]:
    """Picks the resize factor whose detections best match the trained scale.

    Returns (scale, centers_at_that_scale). Runs the model once per candidate
    scale on a centre crop -- not the whole image -- so the search stays cheap
    on large inputs.

    Selection is by how close the detected median tree spacing lands to
    EXPECTED_TREE_SPACING_PX, in log space so that being 2x too zoomed-in and
    2x too zoomed-out are penalised equally. Scales that find too few trees to
    measure are skipped rather than scored, since their spacing is noise.

    KNOWN WEAKNESS -- spacing is a weak signal on its own. The model emits
    blobs roughly 40px apart almost regardless of input, so on imagery whose
    detail has genuinely been destroyed this metric still reports a good
    spacing while the count runs wild (observed: 821 detections scoring the
    best spacing error where the true count was ~100). It reliably picks the
    right scale for a merely-resized photo; it cannot detect that an image is
    too degraded to count, and will return a confident wrong number there.
    Always check the overlay on unfamiliar imagery.
    """
    h, w = image_bgr.shape[:2]

    # Search on a centre crop for speed; big enough to hold many trees.
    probe = 1024
    if h > probe or w > probe:
        y0, x0 = max(0, (h - probe) // 2), max(0, (w - probe) // 2)
        sample = image_bgr[y0:y0 + probe, x0:x0 + probe]
    else:
        sample = image_bgr

    best = (None, None, float("inf"))
    for scale in SCALE_CANDIDATES:
        resized = cv2.resize(
            sample, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        resized = ensure_min_size(resized)
        centers = count_trees(predict_tree_mask(resized, model, device))
        if len(centers) < MIN_DETECTIONS_FOR_SCALE:
            continue
        spacing = _median_spacing(centers)
        if not np.isfinite(spacing) or spacing <= 0:
            continue
        error = abs(np.log(spacing / EXPECTED_TREE_SPACING_PX))
        if error < best[2]:
            best = (scale, centers, error)

    if best[0] is None:
        return 1.0, np.empty((0, 2))  # nothing found at any scale
    return best[0], best[1]


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
    """Draws a ring per detected tree, sized relative to the image.

    Marker and text sizes scale with the image so a small upload does not end
    up as an unreadable mass of oversized circles -- the overlay is the user's
    only way to sanity-check a count, so it has to stay legible at any size.
    """
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]
    radius = max(3, int(round(min(h, w) / 90)))
    thickness = max(1, radius // 5)

    for cx, cy in centers:
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), radius, (0, 0, 255), thickness)

    font_scale = max(0.4, min(h, w) / 500)
    cv2.putText(
        overlay, f"{len(centers)} trees", (12, int(14 + 22 * font_scale)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
        max(3, int(2 * font_scale) + 2), cv2.LINE_AA,
    )
    cv2.putText(
        overlay, f"{len(centers)} trees", (12, int(14 + 22 * font_scale)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
        max(1, int(font_scale)), cv2.LINE_AA,
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
    parser.add_argument(
        "--scale", type=float, default=None,
        help="resize factor before counting. Omit to auto-detect, which is "
             "what makes an arbitrary photo work; pass 1.0 to disable resizing "
             "on imagery already at the source scale.",
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
    model = load_segmentation_model(args.checkpoint, device)

    scale = args.scale
    if scale is None:
        print("Detecting scale...")
        scale, _probe_centers = find_best_scale(masked, model, device)
        print(f"  using scale {scale}x (model expects trees ~{EXPECTED_TREE_SPACING_PX:.0f}px apart)")

    if scale != 1.0:
        masked = cv2.resize(
            masked, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
    masked = ensure_min_size(masked)

    print(f"Stage 2: segmenting on {device}...")
    tree_mask = predict_tree_mask(masked, model, device)
    centers = count_trees(tree_mask)

    # Report positions in the ORIGINAL image's coordinates, so the overlay
    # lines up with what the user uploaded rather than the resized copy.
    if scale != 1.0 and len(centers):
        centers = centers / scale

    print(f"\n{'=' * 40}")
    print(f"TREES COUNTED: {len(centers)}")
    print(f"{'=' * 40}")
    # Deliberately no density figure: with auto-scaling, "per megapixel" is a
    # property of the upload's resolution rather than of the plantation, so it
    # would read as meaningful while comparing nothing across images.

    if args.save_mask:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_mask), tree_mask.astype(np.uint8) * 255)
        print(f"Mask written to {args.save_mask}")

    if args.save_overlay:
        save_overlay(image_bgr, centers, args.save_overlay)
        print(f"Overlay written to {args.save_overlay}")


if __name__ == "__main__":
    main()
