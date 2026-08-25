"""
End-to-end tree counting: raw plantation image -> number of coconut trees.

This is the product path (design spec Goal 1). Everything else in
segmentation/ either builds training data or scores the model against
labeled tiles; this is the only script that takes an arbitrary image and
answers the actual question: how many trees are in it?

Pipeline:
    1. Stage 1 (color MLP) masks out non-vegetation -- sea, buildings,
       soil, roads -- by painting those pixels white, exactly as the
       training data was prepared.
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

CURRENT LIMITATION -- step 1 is not usable. The script that generated
Applicatno/MK-UNet-main/masked images/ is not in the repo, and the
reimplementation here does not reproduce it: end-to-end it undercounts by
~97% (3 trees found where ground truth is 109). So this pipeline currently
requires an already-masked image via --skip-stage1, and refuses to run
Stage 1 rather than print a confidently wrong count. Steps 2-5 ARE
validated: on Amrita_800_1's annotated region the count is 114 against a
ground truth of 109 (+4.6%), matching the +3.2% measured on tiles.

Usage:
    python segmentation/count_trees.py --skip-stage1 \
        --image "Applicatno/MK-UNet-main/masked images/Amrita_800_1.png"

    python segmentation/count_trees.py --skip-stage1 --image img.png \
        --save-overlay counted.png
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

# ---- Stage 1 (color MLP) ----
STAGE1_MODEL = ROOT / "HarinieColourAlgo" / "model" / "mlp_final.keras"
STAGE1_SCALER = ROOT / "HarinieColourAlgo" / "model" / "scaler_final.pkl"

# Stage 1's 4 classes, in the training label order (see
# HarinieColourAlgo/train_model.py). Only vegetation that could be canopy is
# kept; everything else is painted white for Stage 2, which was trained on
# images prepared this way.
# green-800, sea-800, coconut-800 -- i.e. everything EXCEPT non-green-800.
#
# Keeping "sea" looks wrong and is not. Recovered by measuring the reference
# masks in Applicatno/MK-UNet-main/masked images/ directly: of pixels Stage 1
# calls sea, 74.8% are kept there (vs 92.7% green, 98.9% coconut, and only
# 11.5% non-green). Stage 1's sea class is really "dark blue-ish pixel", and
# shadowed canopy between fronds classifies the same way -- so discarding it
# strips out tree interiors, which is exactly the canopy Stage 2 needs.
#
# Cross-check on class populations: {green, sea, coconut} reproduces 39.2% of
# pixels kept against the reference's 42.4%, whereas {green, coconut} gives
# 30.4% -- matching the 30.4% an earlier version of this file actually
# produced, and which undercounted trees by 97%.
STAGE1_KEEP_CLASSES = {0, 2, 3}
STAGE1_WHITE = 255

# A pixel is discarded only when Stage 1 is CONFIDENT it is non-green, rather
# than on a plain argmax. Recovered by matching the reference masks: this
# threshold reproduces their 42.4% kept exactly (argmax gives 39.2%). The
# pixels it rescues are lighter than average canopy (BGR ~66,62,56 vs ~53,47,36)
# -- sunlit fronds and crown edges that argmax narrowly assigns to non-green.
STAGE1_NONGREEN_REJECT_PROB = 0.8

# Half-width of the colour patch Stage 1 characterises each point by. Must
# match PATCH_HALF in HarinieColourAlgo/train_model.py -- the features are
# 5x5 patch statistics, and computing them over any other window changes the
# feature distribution the scaler was fit on.
PATCH_HALF = 2

# Stage 1 classifies a single pixel from its colour statistics, so running it
# per-pixel over an 8192x4283 image means 35M independent MLP calls. Instead
# the image is classified on a coarse grid and the result upsampled -- colour
# regions (sea, buildings, canopy) are far larger than this stride, so the
# boundary error is a few pixels and does not move tree centroids.
STAGE1_STRIDE = 4


def load_segmentation_model(checkpoint_path: Path, device: torch.device):
    model = MK_UNet_ShallowDec(num_classes=2, in_channels=3, enable_cls=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def apply_stage1_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Paints non-vegetation pixels white, reproducing the Stage-1 masked
    images the segmentation model was trained on.

    !! NOT VALIDATED -- DO NOT USE FOR REAL COUNTS. !!

    Reimplementation of Stage 1's dense masking. The script that produced
    Applicatno/MK-UNet-main/masked images/ is not in this repo and predates
    it (those files are dated May 2; the first commit is Aug 22), so this was
    reverse-engineered from the reference masks. It gets close on one image
    and fails on others.

    Progress and where it stands, measured as end-to-end count vs ground
    truth in annotated regions:
      * keep {green, coconut}, argmax     -> Amrita 3 vs 109   (-97%)
      * keep {green, sea, coconut}, argmax-> Amrita 61 vs 109   (-44%)
      * keep unless p(non-green) >= 0.8   -> Amrita 101 vs 109  (-7.3%)
                                             Wat Phleng 1327 vs 2519 (-47%)

    Keeping the "sea" class is correct and counter-intuitive: Stage 1's sea
    class is really "dark blue-ish pixel", and shadowed canopy between fronds
    classifies the same way, so discarding it strips out tree interiors.

    The remaining gap is that no single threshold fits all sites. Reference
    keep-rates vary far more than this reproduction can span:
        Amrita 42.7% (needs ~0.8), Karavatti 80.3% (~0.95),
        Wat Phleng 91.7% (needs >0.99).
    That spread means the original did something this per-pixel MLP pass does
    not capture -- plausibly a morphological/spatial step or a per-image
    adaptive threshold. Fitting the threshold per image is NOT a fix; it just
    launders ground truth into the parameter.

    Beware pixel-agreement metrics here: the -97% version still scored 84%
    pixel agreement and IoU 0.645, because both versions agree about the
    large sea/soil areas. Judge any change by end-to-end count vs ground
    truth on MORE THAN ONE site.
    """
    if not STAGE1_MODEL.exists() or not STAGE1_SCALER.exists():
        raise FileNotFoundError(
            f"Stage 1 model not found at {STAGE1_MODEL}.\n"
            "Pass --skip-stage1 if this image is already Stage-1 masked."
        )

    import joblib
    import tensorflow as tf

    model = tf.keras.models.load_model(str(STAGE1_MODEL))
    scaler = joblib.load(str(STAGE1_SCALER))

    h, w = image_bgr.shape[:2]
    ys = np.arange(0, h, STAGE1_STRIDE)
    xs = np.arange(0, w, STAGE1_STRIDE)

    feats = _stage1_features(image_bgr, ys, xs)
    probs = model.predict(scaler.transform(feats), verbose=0, batch_size=8192)
    nongreen_prob = probs[:, 1].reshape(len(ys), len(xs))
    keep_small = nongreen_prob < STAGE1_NONGREEN_REJECT_PROB

    keep = cv2.resize(
        keep_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    masked = image_bgr.copy()
    masked[~keep] = STAGE1_WHITE
    return masked


def _stage1_features(image_bgr: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """The 14 colour features Stage 1 was trained on, sampled at (ys, xs).

    Mirrors extract_features_for_point() in HarinieColourAlgo/train_model.py,
    which characterises each point by the MEAN and STD of a 5x5 patch around
    it -- not by the single pixel. Computing these from lone pixels instead
    (std forced to 0) puts the features far outside the distribution the
    scaler and MLP were fit on, and Stage 1 then rejects nearly all canopy:
    observed 5.6% of pixels kept versus the reference masks' 42.4%.

    The 5x5 patch statistics are obtained with box filters over the whole
    image, then sampled on the grid -- equivalent to looping per point, but
    vectorised. Feature ORDER must match training exactly; the scaler
    depends on it positionally.
    """
    k = 2 * PATCH_HALF + 1  # 5x5, matching PATCH_HALF=2 in training

    def patch_mean_std(chan: np.ndarray):
        mean = cv2.blur(chan, (k, k), borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.blur(chan * chan, (k, k), borderType=cv2.BORDER_REFLECT)
        var = np.clip(mean_sq - mean * mean, 0.0, None)
        return mean, np.sqrt(var)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    ycr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0

    planes = []
    lab_means, lab_stds = [], []
    for c in range(3):
        mean, std = patch_mean_std(lab[..., c])
        lab_means.append(mean)
        lab_stds.append(std)
    planes.extend(lab_means)
    planes.extend(lab_stds)
    # HSV and YCrCb contribute means only, matching the training feature list.
    planes.extend(cv2.blur(hsv[..., c], (k, k), borderType=cv2.BORDER_REFLECT) for c in range(3))
    planes.extend(cv2.blur(ycr[..., c], (k, k), borderType=cv2.BORDER_REFLECT) for c in range(3))

    # The two RGB ratios come from the CENTRE pixel, not the patch mean.
    b, g, r = (image_bgr[..., i].astype(np.float32) for i in range(3))
    denom = r + g + b + 1e-6
    planes.append(b / denom)
    planes.append(g / denom)

    sampled = [p[np.ix_(ys, xs)] for p in planes]
    return np.stack(sampled, axis=-1).reshape(-1, len(planes))


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
        help="image is already Stage-1 masked (non-vegetation painted white). "
             "Currently REQUIRED -- see --force-unvalidated-stage1.",
    )
    parser.add_argument(
        "--force-unvalidated-stage1", action="store_true",
        help="run the unvalidated Stage 1 reimplementation anyway. It undercounts "
             "by ~97%% on the one image it was checked against; for debugging only.",
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
    elif args.force_unvalidated_stage1:
        print("Stage 1: running UNVALIDATED reimplementation -- count will be wrong.")
        masked = apply_stage1_mask(image_bgr)
        kept = 100.0 * (1.0 - (masked == STAGE1_WHITE).all(2).mean())
        print(f"  kept {kept:.1f}% of pixels as vegetation")
    else:
        # Failing loudly beats printing a confident, badly-wrong number: the
        # reimplementation undercounts by ~97% (see apply_stage1_mask).
        raise SystemExit(
            "Stage 1 masking is not validated: it undercounts by 7% on Amrita "
            "but 47% on Wat Phleng (no single threshold fits both).\n\n"
            "Use an already-masked image with --skip-stage1, e.g.:\n"
            "  python segmentation/count_trees.py --skip-stage1 \\\n"
            "      --image \"Applicatno/MK-UNet-main/masked images/Amrita_800_1.png\"\n\n"
            "Or pass --force-unvalidated-stage1 to run it anyway (debugging only)."
        )

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
