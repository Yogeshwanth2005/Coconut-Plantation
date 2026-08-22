"""
Image & Mask Tiling Script
===========================
Cuts large satellite images and their corresponding masks into smaller tiles,
then splits them into train/val/test folders.

Usage:
    python tile_dataset.py --image_dir ./raw_images --mask_dir ./masks --output_dir ./data/polyp/target/Plantation --tile_size 256

Output structure:
    output_dir/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
"""

import os
import argparse
import random
import numpy as np
import cv2
from PIL import Image


def tile_image_and_mask(image_path, mask_path, tile_size, overlap=0):
    """
    Tiles a large image and its corresponding mask into smaller patches.
    
    Args:
        image_path: Path to the original image
        mask_path: Path to the corresponding mask
        tile_size: Size of each tile (tiles are square)
        overlap: Number of pixels to overlap between tiles
    
    Returns:
        List of (image_tile, mask_tile, tile_name) tuples
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"  ERROR: Could not read image {image_path}")
        return []
    if mask is None:
        print(f"  ERROR: Could not read mask {mask_path}")
        return []
    
    h, w = image.shape[:2]
    mh, mw = mask.shape[:2]
    
    if h != mh or w != mw:
        print(f"  WARNING: Image ({w}x{h}) and mask ({mw}x{mh}) dimensions don't match for {image_path}")
        # Resize mask to match image
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    stride = tile_size - overlap
    tiles = []
    tile_idx = 0
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Ensure tile fits within image (adjust starting position if needed)
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            # Extract tile
            img_tile = image[y_start:y_end, x_start:x_end]
            mask_tile = mask[y_start:y_end, x_start:x_end]
            
            # Skip if tile is smaller than expected (edge case)
            if img_tile.shape[0] != tile_size or img_tile.shape[1] != tile_size:
                continue
            
            tile_name = f"{base_name}_tile_{tile_idx:04d}"
            tiles.append((img_tile, mask_tile, tile_name))
            tile_idx += 1
    
    return tiles


def is_informative_tile(mask_tile, min_foreground_ratio=0.01):
    """
    Checks if a tile contains enough foreground pixels to be useful.
    Returns True if the tile has at least min_foreground_ratio non-zero pixels.
    """
    fg_pixels = np.count_nonzero(mask_tile)
    total_pixels = mask_tile.size
    return (fg_pixels / total_pixels) >= min_foreground_ratio


def main():
    parser = argparse.ArgumentParser(description='Tile images and masks into patches')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with original images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory with mask images')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory (e.g., ./data/polyp/target/Plantation)')
    parser.add_argument('--tile_size', type=int, default=256, help='Tile size in pixels')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between tiles in pixels')
    parser.add_argument('--train_ratio', type=float, default=0.70, help='Training split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--min_fg_ratio', type=float, default=0.0, 
                        help='Minimum foreground ratio to keep a tile (0.0 = keep all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    args = parser.parse_args()
    
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"
    
    random.seed(args.seed)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'masks'), exist_ok=True)
    
    # Find matching image-mask pairs
    image_exts = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(args.mask_dir) 
                  if f.lower().endswith(image_exts)}
    
    all_tiles = []
    
    for img_fname in sorted(os.listdir(args.image_dir)):
        if not img_fname.lower().endswith(image_exts):
            continue
        
        base_name = os.path.splitext(img_fname)[0]
        
        if base_name not in mask_files:
            print(f"  WARNING: No mask for {img_fname}, skipping.")
            continue
        
        image_path = os.path.join(args.image_dir, img_fname)
        mask_path = os.path.join(args.mask_dir, mask_files[base_name])
        
        print(f"Tiling: {img_fname}...")
        tiles = tile_image_and_mask(image_path, mask_path, args.tile_size, args.overlap)
        
        # Filter tiles
        kept = 0
        for img_tile, mask_tile, tile_name in tiles:
            if args.min_fg_ratio > 0 and not is_informative_tile(mask_tile, args.min_fg_ratio):
                continue
            all_tiles.append((img_tile, mask_tile, tile_name))
            kept += 1
        
        print(f"  -> {len(tiles)} tiles generated, {kept} kept")
    
    # Shuffle and split
    random.shuffle(all_tiles)
    n_total = len(all_tiles)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    
    splits = {
        'train': all_tiles[:n_train],
        'val': all_tiles[n_train:n_train + n_val],
        'test': all_tiles[n_train + n_val:]
    }
    
    # Save tiles
    for split_name, split_tiles in splits.items():
        for img_tile, mask_tile, tile_name in split_tiles:
            img_path = os.path.join(args.output_dir, split_name, 'images', f'{tile_name}.png')
            mask_path = os.path.join(args.output_dir, split_name, 'masks', f'{tile_name}.png')
            cv2.imwrite(img_path, img_tile)
            cv2.imwrite(mask_path, mask_tile)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"TILING COMPLETE")
    print(f"{'='*50}")
    print(f"Tile size:  {args.tile_size}×{args.tile_size}")
    print(f"Total tiles: {n_total}")
    print(f"  Train: {len(splits['train'])} ({args.train_ratio*100:.0f}%)")
    print(f"  Val:   {len(splits['val'])} ({args.val_ratio*100:.0f}%)")
    print(f"  Test:  {len(splits['test'])} ({test_ratio*100:.0f}%)")
    print(f"Output:  {args.output_dir}")
    
    # Class distribution in training set
    print(f"\n--- Class Distribution (Training Set) ---")
    all_masks = np.concatenate([m.flatten() for _, m, _ in splits['train']])
    unique, counts = np.unique(all_masks, return_counts=True)
    total = counts.sum()
    for cls_id, count in zip(unique, counts):
        print(f"  Class {cls_id:2d}: {count:>10d} pixels ({count/total*100:5.2f}%)")


if __name__ == '__main__':
    main()
