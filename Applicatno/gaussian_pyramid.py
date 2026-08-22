import os
import cv2 as cv

def generate_multi_scale_pyramid(
    input_root,
    output_root,
    fractions=(4, 8, 16)
):
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    for root, dirs, files in os.walk(input_root):
        current_abs_path = os.path.abspath(root)

        # Skip output directory if nested
        if current_abs_path.startswith(output_root):
            continue

        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            input_path = os.path.join(root, filename)
            img = cv.imread(input_path)

            if img is None:
                print(f"❌ Failed: {filename}")
                continue

            for fraction in fractions:

                # Calculate pyramid levels (log2)
                levels = int(fraction).bit_length() - 1

                img_proc = img.copy()

                # Apply Gaussian pyramid reduction
                for _ in range(levels):
                    img_proc = cv.pyrDown(img_proc)

                # Create output folder name (dataset_1_2 etc.)
                scale_folder = f"dataset_1_{fraction}"

                relative_path = os.path.relpath(root, input_root)
                output_subdir = os.path.join(
                    output_root,
                    scale_folder,
                    relative_path
                )

                os.makedirs(output_subdir, exist_ok=True)

                output_path = os.path.join(output_subdir, filename)
                cv.imwrite(output_path, img_proc)

                print(f"✅ Saved: {output_path}")


input_images_dir = r"D:\Project\Coconut Plantation\Applicatno\images"
output_images_dir = r"D:\Project\Coconut Plantation\Applicatno\Data\MultiScale"

generate_multi_scale_pyramid(
    input_images_dir,
    output_images_dir,
    fractions=(4, 8, 16)
)