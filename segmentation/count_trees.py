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

Scale is handled automatically. The model learned a canopy whose planting
pitch is ~56px, so an upload at a different zoom must be resized first.
find_best_scale() measures that pitch from the image itself by
autocorrelation -- no model involved -- and resizes accordingly. Measured
on a 512x512 plantation crop with 112 known trees:

    native 512px            113  (+0.9%)
    shrunk to 256px, fixed   21  (-81%)   <- what happens without rescaling
    shrunk to 256px, auto   113  (+0.9%)

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


# The model learned trees at the source imagery's scale, so an image at a
# different zoom must be resized before counting -- trees 5px apart are
# invisible to it, trees 200px apart do not resemble its training examples.
#
# Scale is measured from the IMAGE, not from the model's output. Plantations
# are planted on a regular grid, so the canopy has a characteristic spatial
# period which 2D autocorrelation recovers directly. On the training imagery
# that period is ~56.5px; resizing a known-native crop confirms the measure
# is exactly linear in scale (pitch x scale = 57.0/56.0/56.0/56.0 at 1x, 2x,
# 4x, 0.5x), so scale = CANOPY_PITCH_PX / measured_pitch.
#
# An earlier version instead resized until the MODEL's detections sat ~43px
# apart. That failed on real imagery: the model emits blobs roughly 40px
# apart almost regardless of input, so the metric scored best at a scale
# that found 30 trees in a plantation holding several times that. Measuring
# the image sidesteps the model's own bias entirely.
CANOPY_PITCH_PX = 56.5

# Bounds on the correction. Beyond this the imagery is too far from the
# trained scale for a resize to rescue -- upsampling does not restore detail
# that was never captured.
MIN_SCALE, MAX_SCALE = 0.15, 8.0

# Autocorrelation needs enough repeats of the pattern to lock onto it.
MIN_PITCH_PX = 4.0
MAX_PITCH_PX = 200.0


def measure_canopy_pitch(image_bgr: np.ndarray) -> float:
    """Dominant spatial period of the canopy, in pixels, via autocorrelation.

    Returns NaN when no periodicity is found, which is the honest answer for
    imagery that is not a planted grid.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray -= gray.mean()

    spectrum = np.fft.fft2(gray)
    autocorr = np.fft.fftshift(np.fft.ifft2(spectrum * np.conj(spectrum)).real)
    peak = autocorr.max()
    if peak <= 0:
        return float("nan")
    autocorr /= peak

    # Collapse to a radial profile: planting pitch shows up as the first
    # off-centre ridge, whatever direction the rows happen to run.
    cy, cx = np.array(autocorr.shape) // 2
    yy, xx = np.indices(autocorr.shape)
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    counts = np.bincount(radius.ravel())
    profile = np.bincount(radius.ravel(), autocorr.ravel()) / np.maximum(counts, 1)

    # Only trust radii the image can actually support: a period needs several
    # repeats across the frame to be real. Without this bound the profile's
    # noisy far tail wins -- on a 256px crop it reported 181px (against a true
    # 28px) from edge artefacts alone, and the count collapsed to 6.
    limit = min(int(min(autocorr.shape) // 4), int(MAX_PITCH_PX), len(profile))

    # Skip the central autocorrelation peak, then take the first maximum.
    rising = np.diff(profile)
    start = int(np.argmax(rising > 0))
    segment = profile[start:limit]
    if len(segment) < 3:
        return float("nan")

    pitch = float(start + int(np.argmax(segment)))
    return pitch if MIN_PITCH_PX <= pitch <= MAX_PITCH_PX else float("nan")


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


def find_best_scale(image_bgr: np.ndarray) -> float:
    """Resize factor that puts the image at the scale the model was trained on.

    Measured from the image's own canopy periodicity, so this costs one FFT
    and never runs the model. Falls back to 1.0 when no planting pattern is
    detectable, which is the safe default for imagery already at source scale.

    LIMIT -- this corrects zoom, it does not restore detail. An image shrunk
    far below the trained scale has lost the crowns entirely, and upsampling
    it back only feeds the model blur. The MIN_SCALE/MAX_SCALE clamp keeps
    such cases from being silently "fixed" by an extreme resize, but the
    count on genuinely degraded imagery will still be wrong -- check the
    overlay.
    """
    h, w = image_bgr.shape[:2]

    # Measure on a centre crop: large enough for many planting repeats,
    # small enough that the FFT stays cheap on huge source images.
    probe = 1024
    if h > probe or w > probe:
        y0, x0 = max(0, (h - probe) // 2), max(0, (w - probe) // 2)
        sample = image_bgr[y0:y0 + probe, x0:x0 + probe]
    else:
        sample = image_bgr

    pitch = measure_canopy_pitch(sample)
    if not np.isfinite(pitch) or pitch <= 0:
        return 1.0

    return float(np.clip(CANOPY_PITCH_PX / pitch, MIN_SCALE, MAX_SCALE))


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
        pitch = measure_canopy_pitch(masked)
        scale = find_best_scale(masked)
        if np.isfinite(pitch):
            print(f"Scale: canopy pitch {pitch:.0f}px -> resizing {scale:.2f}x "
                  f"(trained on ~{CANOPY_PITCH_PX:.0f}px)")
        else:
            print("Scale: no planting pattern detected, using 1.00x")

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
