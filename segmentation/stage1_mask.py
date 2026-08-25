"""
Stage 1: mask out everything that is not vegetation, so Stage 2 only ever
sees candidate canopy.

WHY THIS EXISTS
The original Stage 1 masking script is not in this repo and predates it
(the masks in Applicatno/MK-UNet-main/masked images/ are dated May 2; the
first commit is Aug 22). Attempts to reverse-engineer it from the colour
MLP in HarinieColourAlgo/ failed: no single per-pixel threshold reproduces
the reference masks across sites (they need ~0.8 on Amrita but >0.99 on
Wat Phleng), and a version scoring 84% pixel agreement still undercounted
trees by 97%.

So this does NOT try to reproduce those masks. That was the wrong goal.
Stage 2 does not need the original mask, it needs *a* mask that removes
non-vegetation. Measured on Amrita_800_1, where Stage 1 discards 57.6% of
the image:

    counting INSIDE annotation boxes   raw 114 vs reference-masked 114
    counting the WHOLE image           raw 3108 vs reference-masked 1613

Inside plantation boxes the mask barely matters -- Stage 2 identifies
canopy from texture on its own. Across a whole image it matters a great
deal: 30.6% of raw detections land on buildings, roads and ornamental
garden trees, which Stage 2 cannot distinguish from coconut palms.

That reframes the requirement. Stage 1 only has to answer "is this pixel
vegetation at all", which is a far easier and more robust question than
"which of the four colour classes is this", and it can be done without
the missing model.

METHOD
Excess Green (ExG = 2G - R - B) is the standard vegetation index for RGB
imagery, used when no near-infrared band is available. Vegetation reflects
strongly in green relative to red and blue; soil, tarmac, concrete and
water do not. The threshold is chosen per image with Otsu's method rather
than fixed, because exposure and haze differ site to site -- that
site-to-site variation is exactly what defeated the fixed-threshold
attempts to rebuild the original.

A morphological opening then removes isolated speckle, and small holes are
filled so shadowed gaps inside a canopy are not punched out of the mask.

RESULTS -- end-to-end tree counts inside annotated regions, which is the
only measure that matters here:

    Amrita_800_1      110 vs GT 109  (+0.9%)   reference Stage 1: +4.6%
    Wat Phleng_800_1 2707 vs GT 2519 (+7.5%)   reference Stage 1: +6.3%

Detections landing on non-vegetation fell from 30.6% (no masking at all)
to 8.6% on Amrita_800_1.

LIMITATIONS -- verified visually, do not assume these are solved:
  * Buildings can survive. Some roofs are green enough to pass ExG; on
    Amrita_800_1's campus corner the large roofs are still kept. This is
    the main residual source of false positives.
  * Ornamental and non-coconut trees are vegetation and are kept by
    design. Separating coconut from other greenery is Stage 2's job, and
    Stage 2 is imperfect at it.
  * The mask is speckled inside dark canopy -- shadowed gaps between
    fronds fall below the threshold and get punched out. Counts survive
    this because Stage 2 works from texture over a 256px tile, not from
    individual pixels, but do not use this mask for area measurements.

Judge changes here by end-to-end tree counts on more than one site, never
by how closely the mask resembles the reference masks and never by mask
IoU -- an earlier rebuild scored 84% pixel agreement while undercounting
trees by 97%.

Usage:
    from stage1_mask import apply_stage1_mask
    masked = apply_stage1_mask(image_bgr)

    python segmentation/stage1_mask.py --image in.png --out masked.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# Pixels rejected as non-vegetation are painted white, matching the
# convention of the reference masked images that Stage 2 was trained on.
WHITE = 255

# Morphology, in pixels at full source resolution (8192x4283). Chosen
# relative to the ~14px blob radius used for tree masks: the opening kernel
# stays well under a single tree so it cannot erase one, while the closing
# kernel is large enough to bridge shadow gaps between fronds of the same
# crown.
OPEN_KERNEL = 3
CLOSE_KERNEL = 9

# Otsu can pick a degenerate threshold on images that are almost entirely
# vegetation (or almost none), because it assumes a bimodal histogram. If
# the resulting mask falls outside these bounds the threshold is clamped to
# a sane fallback rather than accepted.
MIN_KEEP_FRACTION = 0.05
MAX_KEEP_FRACTION = 0.98


def excess_green(image_bgr: np.ndarray) -> np.ndarray:
    """ExG = 2G - R - B on chromatic-normalised channels, as float32.

    Normalising each pixel by its own brightness first makes the index
    depend on colour rather than illumination, so shadowed canopy and
    sunlit canopy score similarly -- important here, because shadowed
    canopy between fronds is exactly what earlier attempts kept losing.
    """
    img = image_bgr.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    total = b + g + r + 1e-6
    return (2.0 * g - r - b) / total


def vegetation_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask, True where the pixel looks like vegetation."""
    exg = excess_green(image_bgr)

    # Otsu needs uint8; map the ExG range onto 0-255 before thresholding.
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _thresh, binary = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    keep = binary.astype(bool)

    # Guard against Otsu misfiring on a near-unimodal histogram (an image
    # that is nearly all canopy, or nearly none).
    fraction = float(keep.mean())
    if not (MIN_KEEP_FRACTION <= fraction <= MAX_KEEP_FRACTION):
        keep = exg > 0.0  # ExG>0 means "greener than neutral", a safe default

    keep_u8 = keep.astype(np.uint8)
    keep_u8 = cv2.morphologyEx(
        keep_u8, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
    )
    keep_u8 = cv2.morphologyEx(
        keep_u8, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    )
    return keep_u8.astype(bool)


def apply_stage1_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Returns the image with non-vegetation painted white."""
    keep = vegetation_mask(image_bgr)
    masked = image_bgr.copy()
    masked[~keep] = WHITE
    return masked


def main():
    parser = argparse.ArgumentParser(description="Mask non-vegetation in an image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Could not read {args.image}")

    masked = apply_stage1_mask(image)
    kept = 100.0 * (1.0 - (masked == WHITE).all(2).mean())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), masked)
    print(f"{args.image.name}: kept {kept:.1f}% as vegetation -> {args.out}")


if __name__ == "__main__":
    main()
