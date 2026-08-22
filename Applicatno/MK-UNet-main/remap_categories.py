"""
Remap JSON Box Categories Based on Original Folder Classification
==================================================================
Scans the Original/ folder (subfolders 1-15 = class IDs) to find which class
each Box ID belongs to, then updates the JSON annotation files with the new
class labels and regenerates masks.

Usage:
    python remap_categories.py --original_dir ./Original --json_dir ./annotations --output_json_dir ./annotations_remapped --output_mask_dir ./generated_masks
"""

import os
import json
import argparse
import numpy as np
import cv2
from collections import defaultdict
from PIL import Image


def build_boxid_to_class(original_dir):
    """
    Scans Original/1/, Original/2/, ..., Original/15/ folders.
    Each image filename (without extension) = Box ID.
    The folder number = class ID.
    
    Returns:
        boxid_to_class: dict mapping Box ID -> class number (1-15)
    """
    boxid_to_class = {}
    
    for folder_name in sorted(os.listdir(original_dir)):
        folder_path = os.path.join(original_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        try:
            class_id = int(folder_name)
        except ValueError:
            continue
        
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                continue
            box_id = os.path.splitext(fname)[0]  # e.g., "AM-1-E-1"
            
            if box_id in boxid_to_class:
                print(f"  WARNING: {box_id} found in both folder {boxid_to_class[box_id]} and {class_id}. Using {class_id}.")
            
            boxid_to_class[box_id] = class_id
    
    return boxid_to_class


def remap_json(json_path, boxid_to_class):
    """
    Reads a JSON annotation file and replaces each box's:
    - Category: with the class number (e.g., "3")
    - Box ID: with "classNumber-lastNumber" (e.g., AM-1-E-2 → "3-2")
    
    Returns:
        remapped_data: list of box dicts with updated Category and Box ID
        stats: dict of class_id -> count of boxes
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    stats = defaultdict(int)
    unmapped = []
    
    for box in data:
        box_id = box['Box ID']
        if box_id in boxid_to_class:
            class_id = boxid_to_class[box_id]
            # Extract the last number from original Box ID (e.g., "AM-1-E-2" → "2")
            last_number = box_id.rsplit('-', 1)[-1]
            # New Box ID: "3-2" (class-lastNumber)
            box['Box ID'] = f"{class_id}-{last_number}"
            box['Category'] = str(class_id)
            stats[class_id] += 1
        else:
            unmapped.append(box_id)
            last_number = box_id.rsplit('-', 1)[-1]
            box['Box ID'] = f"0-{last_number}"
            box['Category'] = '0'  # Assign to background if not found
            stats[0] += 1
    
    return data, dict(stats), unmapped


def create_mask_from_remapped_json(remapped_data, image_shape):
    """
    Creates a mask from remapped JSON data.
    Each bounding box region is filled with its class number (1-15).
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    tree_counts = defaultdict(int)
    
    for box in remapped_data:
        class_id = int(box['Category'])
        
        coords = box['Coordinates']
        x_min = max(0, min(coords['Top-left'][0], w - 1))
        y_min = max(0, min(coords['Top-left'][1], h - 1))
        x_max = max(0, min(coords['Bottom-right'][0], w))
        y_max = max(0, min(coords['Bottom-right'][1], h))
        
        mask[y_min:y_max, x_min:x_max] = class_id
        
        if 'Points' in box:
            tree_counts[class_id] += len(box['Points'])
    
    return mask, dict(tree_counts)


def main():
    parser = argparse.ArgumentParser(description='Remap JSON categories based on Original folder classification')
    parser.add_argument('--original_dir', type=str, default='./Original', help='Path to Original/ folder with class subfolders 1-15')
    parser.add_argument('--json_dir', type=str, default='./annotations', help='Path to JSON annotation files')
    parser.add_argument('--image_dir', type=str, default='./images', help='Path to satellite images')
    parser.add_argument('--output_json_dir', type=str, default='./annotations_remapped', help='Output for remapped JSONs')
    parser.add_argument('--output_mask_dir', type=str, default='./generated_masks_remapped', help='Output for regenerated masks')
    args = parser.parse_args()
    
    os.makedirs(args.output_json_dir, exist_ok=True)
    os.makedirs(args.output_mask_dir, exist_ok=True)
    
    # Step 1: Build Box ID -> Class mapping from Original folders
    print("Scanning Original/ folders...")
    boxid_to_class = build_boxid_to_class(args.original_dir)
    
    # Count per class
    class_counts = defaultdict(int)
    for box_id, cls in boxid_to_class.items():
        class_counts[cls] += 1
    
    print(f"\nFound {len(boxid_to_class)} box-to-class mappings across {len(class_counts)} classes:")
    for cls in sorted(class_counts.keys()):
        print(f"  Class {cls:2d}: {class_counts[cls]:4d} boxes")
    
    # Step 2: Process each JSON file
    print(f"\nRemapping JSON annotations...")
    image_exts = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
    all_unmapped = []
    
    # Save class map (simple 1-15)
    class_map = {str(i): i for i in range(1, 16)}
    class_map_path = os.path.join(args.output_mask_dir, 'class_map.json')
    with open(class_map_path, 'w') as f:
        json.dump(class_map, f, indent=2)
    
    for json_fname in sorted(os.listdir(args.json_dir)):
        if not json_fname.lower().endswith('.json'):
            continue
        
        base_name = os.path.splitext(json_fname)[0]
        json_path = os.path.join(args.json_dir, json_fname)
        
        # Remap categories
        remapped_data, stats, unmapped = remap_json(json_path, boxid_to_class)
        all_unmapped.extend(unmapped)
        
        # Save remapped JSON
        output_json_path = os.path.join(args.output_json_dir, json_fname)
        with open(output_json_path, 'w') as f:
            json.dump(remapped_data, f, indent=4)
        
        # Find matching image for dimensions
        image_path = None
        for ext in image_exts:
            candidate = os.path.join(args.image_dir, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        if image_path:
            with Image.open(image_path) as img:
                w, h = img.size
            
            mask, tree_counts = create_mask_from_remapped_json(remapped_data, (h, w))
            mask_path = os.path.join(args.output_mask_dir, base_name + '.png')
            cv2.imwrite(mask_path, mask)
            
            stats_str = ', '.join([f'cls{k}:{v}' for k, v in sorted(stats.items()) if k > 0])
            print(f"  [{base_name}] {stats_str} | {len(unmapped)} unmapped")
        else:
            print(f"  [{base_name}] No matching image found, skipping mask generation")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"REMAPPING COMPLETE")
    print(f"{'='*60}")
    print(f"Total Box IDs mapped: {len(boxid_to_class)}")
    print(f"Classes: 1-15 + 0 (background)")
    print(f"Remapped JSONs: {args.output_json_dir}")
    print(f"Regenerated masks: {args.output_mask_dir}")
    
    if all_unmapped:
        print(f"\n⚠ {len(all_unmapped)} unmapped Box IDs (assigned to background):")
        for uid in sorted(set(all_unmapped)):
            print(f"    {uid}")
    else:
        print(f"\n✓ All Box IDs successfully mapped!")
    
    print(f"\nNow run:")
    print(f'  python tile_dataset.py --image_dir ./images --mask_dir {args.output_mask_dir} --output_dir ./data/polyp/target/Plantation --tile_size 256')
    print(f'  python train_plantation.py --network MK_UNet_ShallowDec --num_classes 16 --img_size 256')


if __name__ == '__main__':
    main()
