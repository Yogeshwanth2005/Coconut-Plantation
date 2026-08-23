"""
Tiles the Stage-1-masked source images + generated blob masks into
256x256 crops, and splits them into train/val/test with a SITE-AWARE
holdout -- Amrita's tiles are held out entirely for test, never seen
during training. This replaces the old tiling script's approach
(Applicatno/MK-UNet-main/tile_dataset.py), which pooled all sites'
tiles together and randomly shuffled before splitting, making cross-
site generalization untestable (and, per this session's earlier
findings, actively leaky -- sibling crops of the same source tile
could land in both train and test).

Segmentation trains on the Stage-1 color-masked images
(Applicatno/MK-UNet-main/masked images/ -- non-green/non-coconut/sea
areas already blanked white) rather than the raw source photos. This
means the segmentation model never has to learn to distinguish
buildings/roads/sea/soil from canopy -- Stage 1 already solved that --
so it can focus entirely on the harder problem this stage exists for:
which of the green vegetation is specifically coconut canopy.

See docs/superpowers/specs/2026-08-23-tree-count-segmentation-design.md.

Usage:
    python tile_and_split.py

Requires segmentation/masks/ to already exist (run
generate_blob_masks.py first).

Output structure:
    segmentation/tiled/
        train/images/, train/masks/    (Karavatti, Kradangnga, Sambava,
                                         Triple P Kabacan, Wat Phleng --
                                         randomly split 85/15 by tile)
        val/images/, val/masks/
        test/images/, test/masks/      (Amrita only -- held out
                                         entirely from training)
"""

import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MASKED_IMAGES_DIR = ROOT / "Applicatno" / "MK-UNet-main" / "masked images"
MASKS_DIR = Path(__file__).resolve().parent / "masks"
OUTPUT_DIR = Path(__file__).resolve().parent / "tiled"

SITES = [
    "Amrita",
    "Karavatti",
    "Kradangnga",
    "Sambava",
    "Triple P Kabacan",
    "Wat Phleng",
]

HOLDOUT_SITE = "Amrita"  # entire site held out for test, never trained on
VAL_FRACTION = 0.15       # of the non-holdout tiles' crops
TILE_SIZE = 256
SEED = 42


def find_stems_for_site(site: str) -> list[str]:
    """All annotation/mask stems (<Site>_800_<N>) that exist for a site."""
    return sorted(p.stem for p in MASKS_DIR.glob(f"{site}_800_*.png"))


def tile_image_and_mask(image_path: Path, mask_path: Path, tile_size: int):
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(f"Could not read {image_path} or {mask_path}")

    h, w = image.shape[:2]
    base_name = image_path.stem
    tiles = []
    tile_idx = 0

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            img_tile = image[y_start:y_end, x_start:x_end]
            mask_tile = mask[y_start:y_end, x_start:x_end]

            if img_tile.shape[0] != tile_size or img_tile.shape[1] != tile_size:
                continue

            tile_name = f"{base_name}_tile_{tile_idx:04d}"
            tiles.append((img_tile, mask_tile, tile_name))
            tile_idx += 1

    return tiles


def collect_tiles_for_site(site: str) -> list[tuple]:
    all_tiles = []
    for stem in find_stems_for_site(site):
        image_path = MASKED_IMAGES_DIR / f"{stem}.png"
        mask_path = MASKS_DIR / f"{stem}.png"
        if not image_path.exists():
            print(f"  WARNING: no masked image for {stem}, skipping")
            continue
        tiles = tile_image_and_mask(image_path, mask_path, TILE_SIZE)
        all_tiles.extend(tiles)
        print(f"  {stem}: {len(tiles)} tiles")
    return all_tiles


def save_tiles(tiles: list[tuple], split_name: str):
    img_dir = OUTPUT_DIR / split_name / "images"
    mask_dir = OUTPUT_DIR / split_name / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for img_tile, mask_tile, tile_name in tiles:
        cv2.imwrite(str(img_dir / f"{tile_name}.png"), img_tile)
        cv2.imwrite(str(mask_dir / f"{tile_name}.png"), mask_tile)


def main():
    random.seed(SEED)

    if not MASKS_DIR.exists() or not any(MASKS_DIR.glob("*.png")):
        raise FileNotFoundError(
            f"No masks found in {MASKS_DIR} -- run generate_blob_masks.py first."
        )

    print(f"Holdout site (test only, never trained on): {HOLDOUT_SITE}\n")

    print(f"--- Tiling holdout site: {HOLDOUT_SITE} ---")
    test_tiles = collect_tiles_for_site(HOLDOUT_SITE)

    train_val_sites = [s for s in SITES if s != HOLDOUT_SITE]
    print(f"\n--- Tiling train/val sites: {train_val_sites} ---")
    train_val_tiles = []
    for site in train_val_sites:
        print(f" Site: {site}")
        train_val_tiles.extend(collect_tiles_for_site(site))

    random.shuffle(train_val_tiles)
    n_val = int(len(train_val_tiles) * VAL_FRACTION)
    val_tiles = train_val_tiles[:n_val]
    train_tiles = train_val_tiles[n_val:]

    print(f"\nSaving tiles...")
    save_tiles(train_tiles, "train")
    save_tiles(val_tiles, "val")
    save_tiles(test_tiles, "test")

    print(f"\n{'=' * 50}")
    print("TILING + SITE-AWARE SPLIT COMPLETE")
    print(f"{'=' * 50}")
    print(f"Train: {len(train_tiles)} tiles (from {train_val_sites})")
    print(f"Val:   {len(val_tiles)} tiles (from {train_val_sites})")
    print(f"Test:  {len(test_tiles)} tiles (from {HOLDOUT_SITE} ONLY, unseen during training)")

    # Foreground (tree) pixel coverage per split, sanity check
    for split_name, tiles in [("train", train_tiles), ("val", val_tiles), ("test", test_tiles)]:
        if not tiles:
            continue
        fg = sum(np.count_nonzero(m) for _, m, _ in tiles)
        total = sum(m.size for _, m, _ in tiles)
        print(f"  {split_name} tree-pixel coverage: {100 * fg / total:.2f}%")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
