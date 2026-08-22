import os
import json
import cv2

IMAGE_FOLDER = r"D:\Project\Coconut Plantation\Applicatno\dataset_1_8"
JSON_FOLDER = r"D:\Project\Coconut Plantation\Applicatno\scaled_annotations"
OUTPUT_FOLDER = r"D:\Project\Coconut Plantation\Applicatno\Patch Classification"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for img_file in image_files:

    image_path = os.path.join(IMAGE_FOLDER, img_file)
    base_name = os.path.splitext(img_file)[0]
    json_path = os.path.join(JSON_FOLDER, base_name + ".json")

    if not os.path.exists(json_path):
        print(f"⚠ No JSON found for {img_file}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load {image_path}")
        continue

    img_h, img_w = image.shape[:2]

    with open(json_path, "r") as f:
        annotations = json.load(f)

    print(f"\nProcessing {img_file}")

    for entry in annotations:

        box_id = entry["Box ID"]   # 👈 Use exact box name
        coords = entry["Coordinates"]

        x1, y1 = coords["Top-left"]
        x2, y2 = coords["Bottom-right"]

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # Clamp
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        patch = image[y1:y2, x1:x2]

        # 🔥 Save using Box ID only
        out_name = f"{box_id}.png"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)

        cv2.imwrite(out_path, patch)

        print(f"Saved → {out_name}")

print("\n✅ Extraction Complete!")