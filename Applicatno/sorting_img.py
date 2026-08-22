import os
import shutil

DOWNRES_ROOT = r"D:\Project\Coconut Plantation\Applicatno\Patch Classification\Data"
ORIGINAL_ROOT = r"D:\Project\Coconut Plantation\Applicatno\Patch_Data\Original"
OUTPUT_ROOT = r"D:\Project\Coconut Plantation\Applicatno\Patch Classification\Sorted_Images"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 🔹 Index original images
original_files = {}

for root, dirs, files in os.walk(ORIGINAL_ROOT):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            name_only = os.path.splitext(file)[0].lower()
            folder_name = os.path.basename(root)
            original_files[name_only] = folder_name

print("Indexed originals:", len(original_files))


# 🔹 Match and copy DOWNRES images
for root, dirs, files in os.walk(DOWNRES_ROOT):
    for file in files:
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        patch_name = os.path.splitext(file)[0].lower()

        # 🔥 Adjust this based on naming
        base_name = patch_name.split("_patch")[0]

        if base_name in original_files:

            folder_name = original_files[base_name]

            target_folder = os.path.join(OUTPUT_ROOT, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            downres_path = os.path.join(root, file)
            dest_path = os.path.join(target_folder, file)

            shutil.copy2(downres_path, dest_path)

            print("Saved →", folder_name, "/", file)

        else:
            print("No match for:", file)

print("\n✅ Matching Complete!")