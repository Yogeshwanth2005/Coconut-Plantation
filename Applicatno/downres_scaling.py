import os
import json
import cv2

# ==============================
# 🔹 CHANGE THESE FULL PATHS
# ==============================
json_folder = r"D:\Project\Coconut Plantation\Applicatno\annotations"
original_image_folder = r"D:\Project\Coconut Plantation\Applicatno\images"
resized_image_folder = r"D:\Project\Coconut Plantation\Applicatno\dataset_1_8"
output_folder = r"D:\Project\Coconut Plantation\Applicatno\scaled_annotations"
# ==============================

os.makedirs(output_folder, exist_ok=True)

possible_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]

for json_file in os.listdir(json_folder):

    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(json_folder, json_file)

    with open(json_path, "r") as f:
        data = json.load(f)

    base_name = os.path.splitext(json_file)[0]

    # 🔍 Automatically detect correct image extension
    image_name = None
    for ext in possible_extensions:
        test_path = os.path.join(original_image_folder, base_name + ext)
        if os.path.exists(test_path):
            image_name = base_name + ext
            break

    if image_name is None:
        print(f"❌ No matching original image found for {json_file}")
        continue

    original_image_path = os.path.join(original_image_folder, image_name)
    resized_image_path = os.path.join(resized_image_folder, image_name)

    print(f"Processing: {image_name}")

    original_img = cv2.imread(original_image_path)
    resized_img = cv2.imread(resized_image_path)

    if original_img is None:
        print(f"❌ Could not read original image: {original_image_path}")
        continue

    if resized_img is None:
        print(f"❌ Could not read resized image: {resized_image_path}")
        continue

    h1, w1 = original_img.shape[:2]
    h2, w2 = resized_img.shape[:2]

    scale_x = w2 / w1
    scale_y = h2 / h1

    # ======================
    # 🔹 SCALE DATA
    # ======================
    for box in data:

        # Scale box corners
        for key in box["Coordinates"]:
            x, y = box["Coordinates"][key]
            box["Coordinates"][key] = [
                round(x * scale_x, 2),
                round(y * scale_y, 2)
            ]

        # Scale internal points
        box["Points"] = [
            [round(x * scale_x, 2), round(y * scale_y, 2)]
            for x, y in box["Points"]
        ]

        # Recalculate area
        tl = box["Coordinates"]["Top-left"]
        tr = box["Coordinates"]["Top-right"]
        bl = box["Coordinates"]["Bottom-left"]

        width = tr[0] - tl[0]
        height = bl[1] - tl[1]

        box["Area"] = round(width * height, 2)

    # ======================
    # 🔹 SAVE OUTPUT
    # ======================
    output_path = os.path.join(output_folder, json_file)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Saved: {json_file}")

print("\n🎉 All files processed!")