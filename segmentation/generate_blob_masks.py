"""
Generates binary tree/background masks from the existing point
annotations, replacing the old box-fill mask generation
(Applicatno/MK-UNet-main/json_to_masks.py, remap_categories.py) which
filled each annotation's entire bounding box solid -- discarding the
individual tree point locations and producing masks that are just
redrawn rectangles, not tree shapes.

For every annotated tree point, draws a small filled circle onto an
otherwise-blank mask.

Pixels are one of three values:
    0   = background (confirmed not-a-tree)
    255 = tree (an annotated point's blob)
    128 = IGNORE -- unlabeled, excluded from both loss and evaluation

The ignore value exists because annotation boxes cover only ~15% of each
source image on average (as low as 1.8% on Amrita_800_1). Everything
outside a box was never looked at by an annotator, but it is emphatically
NOT background: visual inspection of unannotated regions shows dense,
unmistakable coconut canopy -- and measured across the dataset, 50.0% of
all 256x256 tiles (4468/8928) are >70% Stage-1-kept vegetation with zero
annotation coverage. Training those as background taught the model that
half its examples of "obvious coconut palm" were "not a tree," which is
what produced weak metrics even in-distribution (val IoU 0.21) and made
false-alarm counts uninterpretable (many "FP"s were real, unlabeled
trees). Marking them ignore instead makes the labels honest: a smaller
supervised area, but every supervised pixel is trustworthy.

Note that Stage-1-whitened pixels INSIDE the ignore region stay ignore
too -- being confident it isn't vegetation doesn't make it an annotated
negative, and the supervised background inside the boxes is plentiful.

See docs/superpowers/specs/2026-08-23-tree-count-segmentation-design.md
for the full design and the reasoning behind a fixed (not per-tree
variable) blob radius.

Usage:
    python generate_blob_masks.py

Pairs each JSON in Applicatno/annotations/<Site>_800_<N>.json with its
source tile at Coconut/<SiteFolder>/<Site>_800_<N>.png.

NOTE: "Saved images/<Site> <N>.png" was considered as the image source
but rejected -- those files have annotation dots already baked directly
into the pixels (pre-rendered visualizations), which would leak label
information straight into model input if used for training. The
Coconut/<SiteFolder>/ tiles are confirmed clean (verified visually
against Kradangnga_800_3 during development).

Writes one binary mask PNG per source tile to segmentation/masks/,
same filename stem as the JSON, plus a summary of points drawn/skipped.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = ROOT / "Applicatno" / "annotations"
COCONUT_DIR = ROOT / "Coconut"
OUTPUT_DIR = Path(__file__).resolve().parent / "masks"

SITE_FOLDERS = {
    "Amrita": "Amrita Coimbatore TN",
    "Karavatti": "Karavatti Lakshadweep",
    "Kradangnga": "Kradangnga Thailand",
    "Sambava": "Sambava Madagascar",
    "Triple P Kabacan": "Triple P Kabacan Philippines",
    "Wat Phleng": "Wat Phleng Thailand",
}

# Fixed blob radius in pixels, at full source-tile resolution (8192x4283).
# Derived from the nearest-neighbor spacing distribution across all
# annotated points: p5=36.1px, median=49.7px, mean=53.0px. A radius of
# 14px keeps blobs at roughly a third of the tightest observed spacing,
# so even densely-packed trees get distinct, non-overlapping blobs.
BLOB_RADIUS = 14

# Annotation boxes were deliberately drawn overlapping each other to ensure
# full coverage of each source tile (no gaps between patches) -- a tree
# falling in an overlap region was then marked once per box it appeared in,
# by design, not by mistake. Confirmed on Sambava_800_1: point (4605, 334)
# appears twice at the exact same pixel from two different overlapping
# boxes. Without deduplication this inflates both the mask (merged blobs,
# 1294 points -> 803 actual blobs on that one tile) and any point-count-
# based evaluation. Threshold chosen from the observed duplicate distance
# distribution: true duplicates cluster under ~4px; real distinct trees are
# essentially never this close (nearest-neighbor p5 across all sites is
# 36.1px), so 5px cleanly separates the two without needing manual review.
DEDUP_DISTANCE_PX = 5

# Mask pixel values. IGNORE marks "never annotated -- don't train or
# evaluate here"; the training loss maps it to torch's ignore_index and
# the eval scripts drop it from every metric.
BACKGROUND_VALUE = 0
TREE_VALUE = 255
IGNORE_VALUE = 128


def dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Collapses near-identical points (from overlapping annotation boxes)
    into one, keeping the first occurrence of each cluster."""
    if len(points) < 2:
        return points
    pts = np.array(points)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=DEDUP_DISTANCE_PX)
    to_drop = {max(i, j) for i, j in pairs}
    return [p for idx, p in enumerate(points) if idx not in to_drop]


def find_source_image(json_path: Path) -> Path | None:
    """Pairs Applicatno/annotations/<Site>_800_<N>.json with
    Coconut/<SiteFolder>/<Site>_800_<N>.png (verified clean, no
    pre-rendered annotation dots baked into the pixels)."""
    m = re.match(r"(.+?)_800_(\d+)$", json_path.stem)
    if not m:
        return None
    site = m.group(1)

    folder = SITE_FOLDERS.get(site)
    if folder is None:
        return None

    candidate = COCONUT_DIR / folder / json_path.with_suffix(".png").name
    return candidate if candidate.exists() else None


def generate_mask_for_tile(json_path: Path, image_path: Path) -> tuple[np.ndarray, int, int, int]:
    """Returns (mask, points_drawn, points_out_of_bounds, duplicates_removed).

    Mask is built in three layers, in this order:
      1. Everything starts as IGNORE (nobody annotated it).
      2. Each annotation box's interior becomes BACKGROUND -- an annotator
         did look here, so an absence of points here is a real negative.
      3. Each deduplicated point's blob becomes TREE.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mask = np.full((h, w), IGNORE_VALUE, dtype=np.uint8)

    for box in data:
        coords = box.get("Coordinates")
        if not coords:
            continue
        x0, y0 = coords["Top-left"]
        x1, y1 = coords["Bottom-right"]
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = BACKGROUND_VALUE

    all_points = [tuple(pt) for box in data for pt in box.get("Points", [])]
    deduped_points = dedupe_points(all_points)
    duplicates_removed = len(all_points) - len(deduped_points)

    drawn = 0
    out_of_bounds = 0
    for x, y in deduped_points:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), BLOB_RADIUS, TREE_VALUE, thickness=-1)
            drawn += 1
        else:
            out_of_bounds += 1

    return mask, drawn, out_of_bounds, duplicates_removed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
    print(f"Found {len(json_files)} annotation files in {ANNOTATIONS_DIR}")

    total_drawn = 0
    total_oob = 0
    total_dupes = 0
    unpaired = []

    for jp in json_files:
        image_path = find_source_image(jp)
        if image_path is None:
            unpaired.append(jp.name)
            print(f"  [SKIP] {jp.name}: no matching source image found")
            continue

        mask, drawn, oob, dupes = generate_mask_for_tile(jp, image_path)
        total_drawn += drawn
        total_oob += oob
        total_dupes += dupes

        out_path = OUTPUT_DIR / f"{jp.stem}.png"
        cv2.imwrite(str(out_path), mask)

        tree_pct = 100.0 * (mask == TREE_VALUE).sum() / mask.size
        ignore_pct = 100.0 * (mask == IGNORE_VALUE).sum() / mask.size
        supervised = mask != IGNORE_VALUE
        tree_of_supervised = (
            100.0 * (mask == TREE_VALUE).sum() / supervised.sum() if supervised.any() else 0.0
        )
        print(
            f"  [{jp.stem}] source={image_path.name}  "
            f"points_drawn={drawn}  duplicates_removed={dupes}  out_of_bounds={oob}  "
            f"tree={tree_pct:.2f}%  ignore={ignore_pct:.1f}%  "
            f"tree_within_supervised={tree_of_supervised:.2f}%"
        )

    print(f"\nTotal tree points drawn: {total_drawn}")
    print(f"Total duplicate points removed (overlapping annotation boxes): {total_dupes}")
    if total_oob:
        print(f"Total out-of-bounds points skipped: {total_oob}")
    if unpaired:
        print(f"Unpaired annotation files ({len(unpaired)}): {unpaired}")
    print(f"Masks written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
