# =============================================================================
# pipeline.py
# Coconut Plantation Analysis Pipeline
#
# STAGE 1: Color Classification  @ 1/16 resolution  (MLP model)
# STAGE 2: Patch Classification  @ 1/8  resolution  (ResNet-18 model)
#
# Downsampling: Gaussian Pyramid + Bicubic Interpolation
# =============================================================================

import os
import cv2
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from torchvision import models, transforms
from PIL import Image

# =============================================================================
# >>>  CONFIGURE THESE PATHS BEFORE RUNNING  <<<
# =============================================================================

# Input image (any resolution — pipeline auto-detects)
INPUT_IMAGE = r"D:\Project\Coconut Plantation\Applicatno\images\your_image.jpg"

# Input JSON annotation file for the image (box coordinates + points)
INPUT_JSON  = r"D:\Project\Coconut Plantation\Applicatno\annotations\your_image.json"

# Color MLP model paths
COLOR_MODEL_PATH  = r"D:\Project\Coconut Plantation\HarinieColourAlgo\model\mlp_final.keras"
COLOR_SCALER_PATH = r"D:\Project\Coconut Plantation\HarinieColourAlgo\model\scaler_final.pkl"

# ResNet-18 patch model path
RESNET_MODEL_PATH = r"D:\Project\Coconut Plantation\Applicatno\resnet18_density_classifier.pth"

# Number of density classes the ResNet-18 was trained on (check your training)
NUM_DENSITY_CLASSES = 15

# Output folder
OUTPUT_DIR = r"D:\Project\Coconut Plantation\pipeline_output"

# =============================================================================
# CONSTANTS
# =============================================================================

PATCH_HALF       = 2
NOISY_STD_THRESH = 0.045

LABEL_MAP = {
    "green-800":     0,
    "non-green-800": 1,
    "sea-800":       2,
    "coconut-800":   3,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES  = [IDX_TO_LABEL[i] for i in range(len(LABEL_MAP))]

COLOR_MAP = {
    "green-800":     (0,   200,  0),    # green
    "non-green-800": (200, 200,  0),    # yellow
    "sea-800":       (0,   100, 255),   # blue
    "coconut-800":   (0,   60,  180),   # dark blue
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1 — READ IMAGE & AUTO-DETECT RESOLUTION
# =============================================================================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"❌ Could not load image: {path}")
    H, W = img.shape[:2]
    print(f"✅ Image loaded: {os.path.basename(path)}")
    print(f"   Detected resolution: {W} × {H} px")
    return img, W, H


# =============================================================================
# STEP 2 — GAUSSIAN PYRAMID + BICUBIC DOWNSAMPLING
# =============================================================================

def gaussian_bicubic_downsample(img, scale, label=""):
    """
    Applies Gaussian blur (anti-aliasing) then bicubic resize.
    scale = 1/16, 1/8, etc.
    """
    H, W = img.shape[:2]

    # Determine number of Gaussian pyramid levels
    # e.g. 1/16 = 4 levels,  1/8 = 3 levels
    levels = round(1.0 / scale).bit_length() - 1

    blurred = img.copy()
    for _ in range(levels):
        blurred = cv2.pyrDown(blurred)       # Gaussian pyramid step

    # Final bicubic resize to exact target size
    new_W = max(1, int(W * scale))
    new_H = max(1, int(H * scale))
    resized = cv2.resize(blurred, (new_W, new_H), interpolation=cv2.INTER_CUBIC)

    print(f"   Downsampled ({label}): {W}×{H}  →  {new_W}×{new_H}")
    return resized


# =============================================================================
# STAGE 1 — COLOR CLASSIFICATION  (MLP @ 1/16 resolution)
# =============================================================================

def load_color_model():
    print("\n🔵 Loading Color MLP model...")
    model  = tf.keras.models.load_model(COLOR_MODEL_PATH)
    scaler = joblib.load(COLOR_SCALER_PATH)
    print("   ✅ Color model loaded.")
    return model, scaler


def compute_patch(img, x, y, h=PATCH_HALF):
    H, W = img.shape[:2]
    if x < h or y < h or x >= W - h or y >= H - h:
        return None
    return img[y-h:y+h+1, x-h:x+h+1]


def patch_is_clean(lab):
    return lab.astype(np.float32).std(axis=(0, 1)).max() / 255.0 < NOISY_STD_THRESH


def extract_color_features(img, x, y):
    patch = compute_patch(img, x, y)
    if patch is None:
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    if not patch_is_clean(lab):
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ycr = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

    lab = lab.astype(np.float32) / 255.0
    hsv = hsv.astype(np.float32) / 255.0
    ycr = ycr.astype(np.float32) / 255.0

    b, g, r = patch[PATCH_HALF, PATCH_HALF].astype(np.float32)
    denom = r + g + b + 1e-6

    return np.array([
        lab[..., 0].mean(), lab[..., 1].mean(), lab[..., 2].mean(),
        lab[..., 0].std(),  lab[..., 1].std(),  lab[..., 2].std(),
        hsv[..., 0].mean(), hsv[..., 1].mean(), hsv[..., 2].mean(),
        ycr[..., 0].mean(), ycr[..., 1].mean(), ycr[..., 2].mean(),
        b / denom, g / denom,
    ], dtype=np.float32)


def run_color_classification(img_1_16, scale, annotations, color_model, scaler):
    """
    Runs MLP color model on every annotated point (scaled to 1/16 resolution).
    Returns list of dicts with box_id, point, predicted color label.
    """
    print("\n🟠 STAGE 1: Color Classification @ 1/16 resolution...")
    results = []

    for box in annotations:
        box_id = box.get("Box ID", "unknown")
        points = box.get("Points", [])

        for (px, py) in points:
            # Scale point coordinates to 1/16
            sx = int(px * scale)
            sy = int(py * scale)

            feat = extract_color_features(img_1_16, sx, sy)
            if feat is None:
                continue

            feat_scaled = scaler.transform(feat.reshape(1, -1))
            probs       = color_model.predict(feat_scaled, verbose=0)
            pred_idx    = int(np.argmax(probs))
            label       = IDX_TO_LABEL[pred_idx]

            results.append({
                "box_id":        box_id,
                "orig_x":        px,
                "orig_y":        py,
                "scaled_x":      sx,
                "scaled_y":      sy,
                "color_label":   label,
                "color_conf":    float(np.max(probs)),
            })

    coconut_count = sum(1 for r in results if r["color_label"] == "coconut-800")
    print(f"   ✅ Color classification done. {len(results)} points processed.")
    print(f"   🌴 Coconut points detected: {coconut_count}")
    return results


# =============================================================================
# STAGE 2 — PATCH CLASSIFICATION  (ResNet-18 @ 1/8 resolution)
# =============================================================================

def load_patch_model(num_classes):
    print("\n🔵 Loading ResNet-18 patch model...")

    resnet = models.resnet18(pretrained=False)

    # Match the modified architecture used in training (small image 64×64)
    resnet.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    resnet.fc      = nn.Linear(resnet.fc.in_features, num_classes)

    state = torch.load(RESNET_MODEL_PATH, map_location=DEVICE)
    resnet.load_state_dict(state)
    resnet.to(DEVICE)
    resnet.eval()

    print(f"   ✅ ResNet-18 loaded ({num_classes} density classes).")
    return resnet


PATCH_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def crop_box_patch(img, coords):
    """Crops a bounding box patch from the image."""
    H, W = img.shape[:2]
    tl = coords.get("Top-left",     [0, 0])
    br = coords.get("Bottom-right", [W, H])

    x1, y1 = int(tl[0]), int(tl[1])
    x2, y2 = int(br[0]), int(br[1])

    x1 = max(0, min(W, x1));  x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1));  y2 = max(0, min(H, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def run_patch_classification(img_1_8, scale, annotations, color_results, resnet_model):
    """
    Runs ResNet-18 on every box that contains at least one 'coconut-800' point.
    Operates on 1/8 resolution image.
    Returns list of dicts with box_id, density_class, point_count.
    """
    print("\n🟢 STAGE 2: Patch Classification @ 1/8 resolution...")

    # Collect which box_ids have coconut points
    coconut_boxes = set(
        r["box_id"] for r in color_results if r["color_label"] == "coconut-800"
    )

    results = []

    for box in annotations:
        box_id = box.get("Box ID", "unknown")
        if box_id not in coconut_boxes:
            continue

        coords = box.get("Coordinates", {})
        if not coords:
            continue

        # Scale coordinates to 1/8
        scaled_coords = {}
        for key, val in coords.items():
            scaled_coords[key] = [val[0] * scale, val[1] * scale]

        patch = crop_box_patch(img_1_8, scaled_coords)
        if patch is None or patch.size == 0:
            continue

        # Convert BGR → RGB → PIL for torchvision transforms
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil_patch  = Image.fromarray(patch_rgb)
        tensor     = PATCH_TRANSFORM(pil_patch).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output       = resnet_model(tensor)
            probs        = torch.softmax(output, dim=1)
            pred_class   = int(torch.argmax(probs, dim=1).item())
            confidence   = float(probs[0, pred_class].item())

        num_points = len(box.get("Points", []))

        results.append({
            "box_id":        box_id,
            "density_class": pred_class + 1,     # classes start from 1
            "density_conf":  confidence,
            "num_points":    num_points,
        })

    total_trees = sum(r["num_points"] for r in results)
    print(f"   ✅ Patch classification done. {len(results)} coconut boxes classified.")
    print(f"   🌴 Total annotated tree points in coconut boxes: {total_trees}")
    return results


# =============================================================================
# OUTPUT — SAVE CSV + DENSITY HEATMAP + OVERLAY
# =============================================================================

def save_outputs(orig_img, W, H, color_results, patch_results, img_1_16):
    print("\n📦 Saving outputs...")

    # ── 1. Color predictions CSV ──────────────────────────────────────────────
    color_df = pd.DataFrame(color_results)
    color_csv = os.path.join(OUTPUT_DIR, "color_predictions.csv")
    color_df.to_csv(color_csv, index=False)
    print(f"   💾 Color predictions: {color_csv}")

    # ── 2. Patch predictions CSV ──────────────────────────────────────────────
    patch_df = pd.DataFrame(patch_results)
    patch_csv = os.path.join(OUTPUT_DIR, "patch_predictions.csv")
    patch_df.to_csv(patch_csv, index=False)
    print(f"   💾 Patch predictions: {patch_csv}")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    coconut_pts  = sum(1 for r in color_results if r["color_label"] == "coconut-800")
    total_trees  = sum(r["num_points"] for r in patch_results)
    avg_density  = (sum(r["density_class"] for r in patch_results) / len(patch_results)
                    if patch_results else 0)

    summary = {
        "total_points_color_classified": len(color_results),
        "coconut_points_detected":       coconut_pts,
        "coconut_boxes_classified":      len(patch_results),
        "estimated_total_trees":         total_trees,
        "average_density_class":         round(avg_density, 2),
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"   💾 Summary: {summary_path}")
    print(f"\n   🌴 Estimated total trees: {total_trees}")
    print(f"   📊 Average density class: {avg_density:.1f} / {NUM_DENSITY_CLASSES}")

    # ── 4. Color overlay on 1/16 image ───────────────────────────────────────
    overlay = img_1_16.copy()
    for r in color_results:
        x, y  = r["scaled_x"], r["scaled_y"]
        color = COLOR_MAP.get(r["color_label"], (255, 255, 255))
        cv2.circle(overlay, (x, y), 3, color, -1)

    overlay_path = os.path.join(OUTPUT_DIR, "color_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"   🖼️  Color overlay saved: {overlay_path}")

    # ── 5. Density bar chart ──────────────────────────────────────────────────
    if patch_results:
        classes = [r["density_class"] for r in patch_results]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(classes, bins=range(1, NUM_DENSITY_CLASSES + 2),
                color="#2ecc71", edgecolor="black", align="left")
        ax.set_xlabel("Density Class (1 = sparse → 15 = dense)", fontsize=12)
        ax.set_ylabel("Number of Boxes", fontsize=12)
        ax.set_title("Coconut Patch Density Distribution", fontsize=14)
        ax.set_xticks(range(1, NUM_DENSITY_CLASSES + 1))
        bar_path = os.path.join(OUTPUT_DIR, "density_distribution.png")
        plt.tight_layout()
        plt.savefig(bar_path)
        plt.close()
        print(f"   📊 Density chart saved: {bar_path}")

    print("\n✅ All outputs saved to:", OUTPUT_DIR)
    return summary


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    print("=" * 65)
    print("  🌴 COCONUT PLANTATION ANALYSIS PIPELINE")
    print("=" * 65)

    # ── Load image ─────────────────────────────────────────────────────────
    orig_img, W, H = load_image(INPUT_IMAGE)

    # ── Load annotations ───────────────────────────────────────────────────
    print(f"\n📂 Loading annotations: {os.path.basename(INPUT_JSON)}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"   {len(annotations)} boxes found in annotation file.")

    # ── Gaussian Pyramid + Bicubic Downsampling ────────────────────────────
    print("\n🔺 Generating scaled images...")
    img_1_16 = gaussian_bicubic_downsample(orig_img, scale=1/16, label="1/16 → Color")
    img_1_8  = gaussian_bicubic_downsample(orig_img, scale=1/8,  label="1/8  → Patch")

    # ── Load models ────────────────────────────────────────────────────────
    color_model, scaler = load_color_model()
    resnet_model        = load_patch_model(NUM_DENSITY_CLASSES)

    # ── Stage 1: Color Classification ──────────────────────────────────────
    color_results = run_color_classification(
        img_1_16, scale=1/16,
        annotations=annotations,
        color_model=color_model,
        scaler=scaler,
    )

    # ── Stage 2: Patch Classification ──────────────────────────────────────
    patch_results = run_patch_classification(
        img_1_8, scale=1/8,
        annotations=annotations,
        color_results=color_results,
        resnet_model=resnet_model,
    )

    # ── Save all outputs ───────────────────────────────────────────────────
    summary = save_outputs(orig_img, W, H, color_results, patch_results, img_1_16)

    print("\n" + "=" * 65)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 65)
    return summary


# =============================================================================

if __name__ == "__main__":
    run_pipeline()
