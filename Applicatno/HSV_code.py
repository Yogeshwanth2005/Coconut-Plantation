import os
import cv2
import numpy as np

INPUT_FOLDER = r"Data_Green_NonGreen"
OUTPUT_FOLDER = r"Data_Green_NonGreen/HS"

# Pixel threshold above which tiled processing is used to avoid OOM
LARGE_IMAGE_THRESHOLD = 50_000_000  # 50 megapixels

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

processed = 0
failed = 0

# Collect all images to process (handles both flat and nested structures)
images_to_process = []

for item in os.listdir(INPUT_FOLDER):
    item_path = os.path.join(INPUT_FOLDER, item)
    
    if os.path.isdir(item_path):
        # Found a subdirectory (nested structure)
        save_class_path = os.path.join(OUTPUT_FOLDER, item)
        os.makedirs(save_class_path, exist_ok=True)
        for img_name in os.listdir(item_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                images_to_process.append((img_name, os.path.join(item_path, img_name), save_class_path))
    elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Found an image directly in the input folder (flat structure)
        images_to_process.append((item, item_path, OUTPUT_FOLDER))

for img_name, img_path, save_dir in images_to_process:
    # Use IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION so EXIF rotation
    # does not silently change the image dimensions on some builds.
    img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if img is None:
        print(f"  [FAILED]  {img_path}")
        failed += 1
        continue

    h, w, c = img.shape
    total_pixels = h * w
    print(f"  Processing: {img_name}  |  {w}x{h}  ({total_pixels:,} px)")

    save_path = os.path.join(save_dir, img_name)

    if total_pixels > LARGE_IMAGE_THRESHOLD:
        # --- Tile-based processing for very large / high-resolution images ---
        # Process in horizontal strips to keep peak RAM usage manageable.
        TILE_ROWS = 1024  # rows per tile; tune as needed
        hs_image = np.empty((h, w, 3), dtype=np.uint8)

        for row_start in range(0, h, TILE_ROWS):
            row_end = min(row_start + TILE_ROWS, h)
            tile_bgr = img[row_start:row_end, :, :]
            tile_hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
            Ht = tile_hsv[:, :, 0]
            St = tile_hsv[:, :, 1]
            hs_image[row_start:row_end, :, :] = cv2.merge([Ht, St, Ht])
            del tile_bgr, tile_hsv, Ht, St
    else:
        # --- Standard in-memory processing for normal-resolution images ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        hs_image = cv2.merge([H, S, H])
        del hsv, H, S

    cv2.imwrite(save_path, hs_image)
    del img, hs_image
    processed += 1

print(f"\nDone! ✅  Processed: {processed}  |  Failed: {failed}")