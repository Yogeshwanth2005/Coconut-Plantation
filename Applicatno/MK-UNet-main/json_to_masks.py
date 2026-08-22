"""
JSON Annotation to Segmentation Mask Converter
================================================
Reads JSON annotation files (bounding boxes with category labels and tree points)
and generates:
1. Grayscale mask images where each pixel = class ID
2. A class_map.json file mapping category names to class IDs
3. A tree_counts.csv summary with tree counts per class per image

Usage:
    python json_to_masks.py --image_dir ./raw_images --json_dir ./raw_json --output_dir ./masks
"""

import os
import json
import csv
import argparse
import numpy as np
import cv2
from collections import defaultdict
from PIL import Image


def discover_categories(json_dir):
    """Scan all JSON files and extract unique category names."""
    categories = set()
    for fname in os.listdir(json_dir):
        if not fname.lower().endswith('.json'):
            continue
        with open(os.path.join(json_dir, fname), 'r') as f:
            data = json.load(f)
        for box in data:
            categories.add(box['Category'])
    
    # Sort for deterministic class IDs
    sorted_cats = sorted(categories)
    # 0 = Background, 1..N = plantation types
    class_map = {cat: idx + 1 for idx, cat in enumerate(sorted_cats)}
    return class_map


def create_mask_from_json(json_path, image_shape, class_map):
    """
    Create a segmentation mask from a JSON annotation file.
    
    Args:
        json_path: Path to JSON file
        image_shape: (height, width) of the corresponding image
        class_map: dict mapping category name -> class ID
    
    Returns:
        mask: numpy array [H, W] with class IDs
        tree_counts: dict mapping class_name -> count of trees
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    tree_counts = defaultdict(int)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for box in data:
        category = box['Category']
        class_id = class_map.get(category, 0)
        
        # Get bounding box coordinates
        coords = box['Coordinates']
        x_min = coords['Top-left'][0]
        y_min = coords['Top-left'][1]
        x_max = coords['Bottom-right'][0]
        y_max = coords['Bottom-right'][1]
        
        # Clamp to image bounds
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(0, min(x_max, w))
        y_max = max(0, min(y_max, h))
        
        # Fill bounding box region with class ID
        mask[y_min:y_max, x_min:x_max] = class_id
        
        # Count trees (points)
        if 'Points' in box:
            tree_counts[category] += len(box['Points'])
    
    return mask, dict(tree_counts)


def main():
    parser = argparse.ArgumentParser(description='Convert JSON annotations to segmentation masks')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing original images')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing JSON annotation files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for masks')
    parser.add_argument('--class_map_file', type=str, default=None, help='Optional: path to existing class_map.json')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Discover or load class mapping
    if args.class_map_file and os.path.exists(args.class_map_file):
        with open(args.class_map_file, 'r') as f:
            class_map = json.load(f)
        print(f"Loaded class map from {args.class_map_file}")
    else:
        class_map = discover_categories(args.json_dir)
        # Save the class map
        class_map_path = os.path.join(args.output_dir, 'class_map.json')
        with open(class_map_path, 'w') as f:
            json.dump(class_map, f, indent=2)
        print(f"Discovered {len(class_map)} categories. Saved class map to {class_map_path}")
    
    # Print class mapping
    print("\n--- Class Mapping ---")
    print(f"  0 : Background")
    for cat, idx in sorted(class_map.items(), key=lambda x: x[1]):
        print(f"  {idx} : {cat}")
    print()
    
    # Step 2: Process each JSON file
    image_exts = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
    all_tree_counts = []
    processed = 0
    skipped = 0
    
    for json_fname in sorted(os.listdir(args.json_dir)):
        if not json_fname.lower().endswith('.json'):
            continue
        
        base_name = os.path.splitext(json_fname)[0]
        
        # Find matching image
        image_path = None
        for ext in image_exts:
            candidate = os.path.join(args.image_dir, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        if image_path is None:
            print(f"  WARNING: No matching image found for {json_fname}, skipping.")
            skipped += 1
            continue
        
        # Get image dimensions
        with Image.open(image_path) as img:
            w, h = img.size  # PIL returns (width, height)
        
        # Create mask
        json_path = os.path.join(args.json_dir, json_fname)
        mask, tree_counts = create_mask_from_json(json_path, (h, w), class_map)
        
        # Save mask as grayscale PNG
        mask_path = os.path.join(args.output_dir, base_name + '.png')
        cv2.imwrite(mask_path, mask)
        
        # Record tree counts
        row = {'image': base_name, 'total_trees': sum(tree_counts.values())}
        for cat in sorted(class_map.keys()):
            row[f'trees_{cat}'] = tree_counts.get(cat, 0)
        all_tree_counts.append(row)
        
        processed += 1
        print(f"  [{processed}] {json_fname} -> {mask_path} ({sum(tree_counts.values())} trees)")
    
    # Step 3: Save tree count summary
    if all_tree_counts:
        csv_path = os.path.join(args.output_dir, 'tree_counts.csv')
        keys = all_tree_counts[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_tree_counts)
        print(f"\nTree count summary saved to {csv_path}")
    
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Masks saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
