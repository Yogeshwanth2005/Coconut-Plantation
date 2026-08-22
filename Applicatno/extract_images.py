import os
import json
import cv2
from pathlib import Path

# ---------- CONFIG ----------
IMAGE_FOLDER = "images"          # .png/.jpg images
JSON_FOLDER  = "annotations"     # labeled JSONs with Box ID / Category
OUTPUT_FOLDER = "patches"        # where cropped patches will be saved
DRAW_POINTS = True               # draw Points onto the patch
# -----------------------------

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

def clamp_coords(image, x1, y1, x2, y2):
    H, W = image.shape[:2]
    x1 = max(0, min(W, int(x1))); x2 = max(0, min(W, int(x2)))
    y1 = max(0, min(H, int(y1))); y2 = max(0, min(H, int(y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def unique_path(base_dir: Path, name: str, ext: str = ".png") -> Path:
    p = base_dir / f"{name}{ext}"
    k = 2
    while p.exists():
        p = base_dir / f"{name}_{k}{ext}"
        k += 1
    return p

def bin_folder_by_points(n: int) -> str:
    """Return folder name based on number of trees (Points length)."""
    if n == 0:
        return "00"                 # no trees
    lo = ((n - 1) // 10) * 10 + 1   # 1,11,21,...
    hi = lo + 9                     # 10,20,30,...
    # cap naming above 100 as '101+'
    if hi <= 100:
        return f"{lo:02d}-{hi:02d}"
    else:
        return "101+"

def process_image(image_path: Path, json_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️ Could not load image: {image_path.name}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to read JSON {json_path.name}: {e}")
        return

    if not isinstance(data, list):
        print(f"⚠️ JSON {json_path.name} is not a list — skipping.")
        return

    saved = 0
    for idx, entry in enumerate(data):
        coords = entry.get("Coordinates")
        if not coords or "Top-left" not in coords or "Bottom-right" not in coords:
            continue

        box_id = entry.get("Box ID")
        if not box_id or not isinstance(box_id, str) or not box_id.strip():
            # If you prefer a fallback: box_id = f"{image_path.stem}-{idx+1}"
            continue

        x1, y1 = coords["Top-left"]
        x2, y2 = coords["Bottom-right"]
        x1, y1, x2, y2 = clamp_coords(image, x1, y1, x2, y2)

        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            print(f"⚠️ Empty patch for {box_id} in {image_path.name}")
            continue

        # decide folder by number of trees
        num_points = len(entry.get("Points", []))
        bin_folder = bin_folder_by_points(num_points)

        if DRAW_POINTS:
            for (px, py) in entry.get("Points", []):
                px_local, py_local = int(px - x1), int(py - y1)
                if 0 <= px_local < patch.shape[1] and 0 <= py_local < patch.shape[0]:
                    cv2.circle(patch, (px_local, py_local), 4, (0, 0, 255), -1)

        out_dir = Path(OUTPUT_FOLDER) / bin_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = unique_path(out_dir, box_id, ".png")
        cv2.imwrite(str(out_path), patch)
        saved += 1

    print(f"✅ {image_path.name}: saved {saved} patches")

def main():
    images = [f for f in os.listdir(IMAGE_FOLDER)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_file in images:
        stem = os.path.splitext(img_file)[0]
        json_path = Path(JSON_FOLDER) / f"{stem}.json"
        if not json_path.exists():
            print(f"⚠️ No JSON found for {img_file}, skipping…")
            continue
        process_image(Path(IMAGE_FOLDER) / img_file, json_path)

if __name__ == "__main__":
    main()
