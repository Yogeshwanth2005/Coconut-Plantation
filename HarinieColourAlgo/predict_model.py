# predict_model.py
# Point-level prediction + metrics for color-only MLP model
# Outputs CSV + confusion matrix image in SAME folder

import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# CONFIG
# =========================================================
CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "predict_images"
MODEL_DIR = "model"

OUT_CSV = "point_predictions.csv"
CONF_MATRIX_IMG = "prediction_confusion_matrix.png"

PATCH_HALF = 2
NOISY_STD_THRESH = 0.045

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [IDX_TO_LABEL[i] for i in range(len(LABEL_MAP))]

# =========================================================
# LABEL NORMALIZATION (SAME AS TRAIN)
# =========================================================
def normalize_label(s):
    s = str(s).strip().lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")

    if "nongreen" in s:
        return "non-green-800"
    if "green800" in s:
        return "green-800"
    if "sea800" in s or s == "sea":
        return "sea-800"
    if "coconut" in s:
        return "coconut-800"
    return None

# =========================================================
# FEATURE EXTRACTION (IDENTICAL TO TRAIN)
# =========================================================
def compute_patch(img, x, y, h=PATCH_HALF):
    H, W = img.shape[:2]
    if x < h or y < h or x >= W - h or y >= H - h:
        return None
    return img[y-h:y+h+1, x-h:x+h+1]

def patch_is_clean(lab):
    return lab.astype(np.float32).std(axis=(0,1)).max() / 255.0 < NOISY_STD_THRESH

def extract_features_for_point(img, x, y):
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
        lab[...,0].mean(), lab[...,1].mean(), lab[...,2].mean(),
        lab[...,0].std(),  lab[...,1].std(),  lab[...,2].std(),
        hsv[...,0].mean(), hsv[...,1].mean(), hsv[...,2].mean(),
        ycr[...,0].mean(), ycr[...,1].mean(), ycr[...,2].mean(),
        b/denom, g/denom
    ], dtype=np.float32)

# =========================================================
# LOAD MODEL + SCALER
# =========================================================
print("Loading model and scaler...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "mlp_final.keras"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_final.pkl"))

# =========================================================
# LOAD CSV
# =========================================================
df = pd.read_csv(CSV_FILE)
df["label"] = df["label"].apply(normalize_label)
df = df.dropna(subset=["label", "points"])

records = []

# =========================================================
# PREDICTION LOOP (POINT-LEVEL)
# =========================================================
for _, row in df.iterrows():
    filename = row["filename"]
    expected = row["label"]

    img_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    for pt in row["points"].split(";"):
        try:
            x, y = map(int, map(float, pt.split(",")))
        except:
            continue

        feat = extract_features_for_point(img, x, y)
        if feat is None:
            continue

        feat_scaled = scaler.transform(feat.reshape(1, -1))
        probs = model.predict(feat_scaled, verbose=0)
        pred_idx = int(np.argmax(probs))
        predicted = IDX_TO_LABEL[pred_idx]

        result = "correct" if predicted == expected else "wrong"

        records.append({
            "filename": filename,
            "x": x,
            "y": y,
            "expected": expected,
            "predicted": predicted,
            "result": result
        })

# =========================================================
# SAVE CSV
# =========================================================
out_df = pd.DataFrame(records)
out_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved predictions to: {OUT_CSV}")
print(f"Total predicted points: {len(out_df)}")

# =========================================================
# METRICS + CONFUSION MATRIX (POINT-LEVEL)
# =========================================================
if len(out_df) > 0:
    y_true = out_df["expected"].map(LABEL_MAP).values
    y_pred = out_df["predicted"].map(LABEL_MAP).values

    acc = accuracy_score(y_true, y_pred)
    print(f"\nPOINT-LEVEL ACCURACY: {acc:.4f}")

    print("\nCLASSIFICATION REPORT:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)

    print("\nCONFUSION MATRIX (True rows × Predicted columns):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Point-Level Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_IMG)
    plt.close()

    print(f"\nSaved confusion matrix image to: {CONF_MATRIX_IMG}")
else:
    print("\n❌ No valid points predicted — metrics skipped.")




























'''# predict_model.py
# Point-level prediction + metrics for color-only MLP model

import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# CONFIG
# =========================================================
CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "predict_images"
MODEL_DIR = "model"
OUT_CSV = "point_predictions.csv"

PATCH_HALF = 2
NOISY_STD_THRESH = 0.045

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [IDX_TO_LABEL[i] for i in range(len(LABEL_MAP))]

# =========================================================
# LABEL NORMALIZATION (SAME AS TRAIN)
# =========================================================
def normalize_label(s):
    s = str(s).strip().lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")

    if "nongreen" in s:
        return "non-green-800"
    if "green800" in s:
        return "green-800"
    if "sea800" in s or s == "sea":
        return "sea-800"
    if "coconut" in s:
        return "coconut-800"
    return None

# =========================================================
# FEATURE EXTRACTION (IDENTICAL TO TRAIN)
# =========================================================
def compute_patch(img, x, y, h=PATCH_HALF):
    H, W = img.shape[:2]
    if x < h or y < h or x >= W - h or y >= H - h:
        return None
    return img[y-h:y+h+1, x-h:x+h+1]

def patch_is_clean(lab):
    return lab.astype(np.float32).std(axis=(0,1)).max() / 255.0 < NOISY_STD_THRESH

def extract_features_for_point(img, x, y):
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
        lab[...,0].mean(), lab[...,1].mean(), lab[...,2].mean(),
        lab[...,0].std(),  lab[...,1].std(),  lab[...,2].std(),
        hsv[...,0].mean(), hsv[...,1].mean(), hsv[...,2].mean(),
        ycr[...,0].mean(), ycr[...,1].mean(), ycr[...,2].mean(),
        b/denom, g/denom
    ], dtype=np.float32)

# =========================================================
# LOAD MODEL + SCALER
# =========================================================
print("Loading model and scaler...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "mlp_final.keras"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_final.pkl"))

# =========================================================
# LOAD CSV
# =========================================================
df = pd.read_csv(CSV_FILE)
df["label"] = df["label"].apply(normalize_label)
df = df.dropna(subset=["label", "points"])

records = []

# =========================================================
# PREDICTION LOOP (POINT-LEVEL)
# =========================================================
for _, row in df.iterrows():
    filename = row["filename"]
    expected = row["label"]

    img_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    for pt in row["points"].split(";"):
        try:
            x, y = map(int, map(float, pt.split(",")))
        except:
            continue

        feat = extract_features_for_point(img, x, y)
        if feat is None:
            continue

        feat_scaled = scaler.transform(feat.reshape(1, -1))
        probs = model.predict(feat_scaled, verbose=0)
        pred_idx = int(np.argmax(probs))
        predicted = IDX_TO_LABEL[pred_idx]

        result = "correct" if predicted == expected else "wrong"

        records.append({
            "filename": filename,
            "x": x,
            "y": y,
            "expected": expected,
            "predicted": predicted,
            "result": result
        })

# =========================================================
# SAVE CSV
# =========================================================
out_df = pd.DataFrame(records)
out_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved predictions to: {OUT_CSV}")
print(f"Total predicted points: {len(out_df)}")

# =========================================================
# METRICS (POINT-LEVEL)
# =========================================================
if len(out_df) > 0:
    y_true = out_df["expected"].map(LABEL_MAP).values
    y_pred = out_df["predicted"].map(LABEL_MAP).values

    acc = accuracy_score(y_true, y_pred)
    print(f"\nPOINT-LEVEL ACCURACY: {acc:.4f}")

    print("\nCLASSIFICATION REPORT:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)

    print("\nCONFUSION MATRIX (True rows × Predicted columns):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Point-Level Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "prediction_confusion_matrix.png"))
    plt.close()
else:
    print("\n❌ No valid points predicted — metrics skipped.")

'''



























