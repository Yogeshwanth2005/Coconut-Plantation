"""
Generates binary tree/background masks from the existing point
annotations, replacing the old box-fill mask generation
(Applicatno/MK-UNet-main/json_to_masks.py, remap_categories.py) which
filled each annotation's entire bounding box solid -- discarding the
individual tree point locations and producing masks that are just
redrawn rectangles, not tree shapes.

For every annotated tree point, draws a small filled circle onto an
otherwise-blank mask. Everything else is background (0).

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


def generate_mask_for_tile(json_path: Path, image_path: Path) -> tuple[np.ndarray, int, int]:
    """Returns (mask, points_drawn, points_out_of_bounds)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    drawn = 0
    out_of_bounds = 0
    for box in data:
        for x, y in box.get("Points", []):
            x, y = int(round(x)), int(round(y))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(mask, (x, y), BLOB_RADIUS, 255, thickness=-1)
                drawn += 1
            else:
                out_of_bounds += 1

    return mask, drawn, out_of_bounds


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
    print(f"Found {len(json_files)} annotation files in {ANNOTATIONS_DIR}")

    total_drawn = 0
    total_oob = 0
    unpaired = []

    for jp in json_files:
        image_path = find_source_image(jp)
        if image_path is None:
            unpaired.append(jp.name)
            print(f"  [SKIP] {jp.name}: no matching source image found")
            continue

        mask, drawn, oob = generate_mask_for_tile(jp, image_path)
        total_drawn += drawn
        total_oob += oob

        out_path = OUTPUT_DIR / f"{jp.stem}.png"
        cv2.imwrite(str(out_path), mask)

        coverage_pct = 100.0 * np.count_nonzero(mask) / mask.size
        print(
            f"  [{jp.stem}] source={image_path.name}  "
            f"points_drawn={drawn}  out_of_bounds={oob}  "
            f"tree_pixel_coverage={coverage_pct:.2f}%"
        )

    print(f"\nTotal tree points drawn: {total_drawn}")
    if total_oob:
        print(f"Total out-of-bounds points skipped: {total_oob}")
    if unpaired:
        print(f"Unpaired annotation files ({len(unpaired)}): {unpaired}")
    print(f"Masks written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
