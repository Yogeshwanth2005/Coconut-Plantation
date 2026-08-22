# train_model.py
# Color-only training pipeline with MLP classifier
# Uses handcrafted color statistics (NO texture / NO spatial DL features)

import os
import cv2
import random
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# =========================================================
# CONFIG
# =========================================================
RND = 42
random.seed(RND)
np.random.seed(RND)
tf.random.set_seed(RND)

CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "train_images"      # <-- TRAIN IMAGES ONLY
MODEL_DIR = "model"

PATCH_HALF = 2
NOISY_STD_THRESH = 0.045
BATCH_SIZE = 64
EPOCHS = 300

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

NUM_CLASSES = len(LABEL_MAP)
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# LABEL NORMALIZATION
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
# CSV PREPROCESSING
# =========================================================
def preprocess_csv(csv_path, image_folder):
    df = pd.read_csv(csv_path)

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])

    def clean_points(p):
        p = str(p).strip()
        if not p:
            return None
        pts = []
        for t in p.split(";"):
            if "," in t:
                try:
                    x, y = map(float, t.split(","))
                    pts.append(f"{x},{y}")
                except:
                    pass
        return ";".join(pts) if pts else None

    df["points"] = df["points"].apply(clean_points)
    df = df.dropna(subset=["points"])

    df["filepath"] = df["filename"].apply(
        lambda f: os.path.join(image_folder, str(f))
    )
    df = df[df["filepath"].apply(os.path.exists)]

    return df.reset_index(drop=True)

# =========================================================
# FEATURE EXTRACTION (COLOR-ONLY — UNCHANGED)
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
# LOAD FEATURES
# =========================================================
def load_features():
    df = preprocess_csv(CSV_FILE, IMAGE_FOLDER)

    X, y = [], []

    for _, row in df.iterrows():
        img = cv2.imread(row["filepath"])
        label = LABEL_MAP[row["label"]]

        for pt in row["points"].split(";"):
            x, y_pt = map(int, map(float, pt.split(",")))
            feat = extract_features_for_point(img, x, y_pt)
            if feat is not None:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y)

# =========================================================
# MODEL (MLP = DL CLASSIFIER, NOT FEATURE EXTRACTOR)
# =========================================================
def build_mlp(in_dim):
    model = models.Sequential([
        layers.Input(shape=(in_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading features...")
    X, y = load_features()

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for label_idx, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[label_idx]}: {count} samples")
    
    # If all classes have at least 2 samples, use stratified split
    # Otherwise, use regular split
    if counts.min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y,
            test_size=0.2,
            random_state=RND
        )
    else:
        print("\n⚠ Warning: Some classes have < 2 samples, using non-stratified split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RND
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_final.pkl"))

    cw_vals = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(cw_vals))

    model = build_mlp(X_train_s.shape[1])

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", patience=7, factor=0.5),
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    ]

    model.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    model.save(os.path.join(MODEL_DIR, "mlp_final.keras"))

    preds = np.argmax(model.predict(X_test_s), axis=1)

    print("\nFINAL ACCURACY:", accuracy_score(y_test, preds))
    print("\nCLASSIFICATION REPORT:\n")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, preds)
    print("\nCONFUSION MATRIX:")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_labeled.png"))
    plt.close()

    for i, cname in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        print(f"\n▶ Class: {cname}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")

        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                print(f"    {cname} → {CLASS_NAMES[j]} : {cm[i,j]}")

if __name__ == "__main__":
    main()













'''# train_color_final_hard_guarantee.py
"""
Full training pipeline (Option A - Hard Guarantee):
- CSV preprocessing
- Color-only feature extraction (5x5 patch)
- Augmentation + oversampling
- Class-guaranteed image-level split (duplicates allowed when necessary)
- MLP + RandomForest + KNN ensemble
- RF/KNN probability expansion to full class vector
- Always prints 4-class classification_report safely

MODIFIED:
✔ Labeled confusion matrix (printed + saved as image)
✔ Explicit True Positives, False Positives, False Negatives
✔ Detailed class-to-class misclassification report
"""

import os
import cv2
import random
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ---------------- CONFIG ----------------
RND = 42
random.seed(RND)
np.random.seed(RND)
tf.random.set_seed(RND)

CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "images"
MODEL_DIR = "model"

PATCH_HALF = 2
NOISY_STD_THRESH = 0.045
AUGMENT_COPIES = 2
BATCH_SIZE = 64
EPOCHS = 300

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

NUM_CLASSES = len(LABEL_MAP)
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================================
# LABEL NORMALIZATION
# =====================================================================
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

# =====================================================================
# CSV PREPROCESSING
# =====================================================================
def preprocess_csv(csv_path, image_folder):
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])

    def clean_points(p):
        p = str(p).strip()
        if not p:
            return None
        pts = []
        for t in p.split(";"):
            if "," in t:
                try:
                    x, y = map(float, t.split(","))
                    pts.append(f"{x},{y}")
                except:
                    pass
        return ";".join(pts) if pts else None

    df["points"] = df["points"].apply(clean_points)
    df = df.dropna(subset=["points"])

    df["filepath"] = df["filename"].apply(
        lambda f: os.path.join(image_folder, str(f))
    )
    df = df[df["filepath"].apply(os.path.exists)]

    return df.reset_index(drop=True)

# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
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

# =====================================================================
# LOAD FEATURES
# =====================================================================
def load_grouped_features():
    df = preprocess_csv(CSV_FILE, IMAGE_FOLDER)
    grouped = defaultdict(list)

    for _, row in df.iterrows():
        img = cv2.imread(row["filepath"])
        label = LABEL_MAP[row["label"]]

        for pt in row["points"].split(";"):
            x, y = map(int, map(float, pt.split(",")))
            feat = extract_features_for_point(img, x, y)
            if feat is not None:
                grouped[row["filename"]].append((feat, label))

    return grouped

# =====================================================================
# MODEL
# =====================================================================
def build_mlp(in_dim):
    m = models.Sequential([
        layers.Input(shape=(in_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m

# =====================================================================
# MAIN
# =====================================================================
def main():
    grouped = load_grouped_features()

    X, y = [], []
    for items in grouped.values():
        for f, l in items:
            X.append(f)
            y.append(l)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RND
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    mlp = build_mlp(X_train_s.shape[1])
    mlp.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[ReduceLROnPlateau(monitor="val_loss", patience=7)]
    )

    preds = np.argmax(mlp.predict(X_test_s), axis=1)

    print("\nENSEMBLE ACCURACY:", accuracy_score(y_test, preds))
    print("\nCLASSIFICATION REPORT:\n")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

    # ===============================================================
    # CONFUSION MATRIX + ANALYSIS
    # ===============================================================
    cm = confusion_matrix(y_test, preds)

    print("\nCONFUSION MATRIX (True rows × Predicted columns):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Labeled)")
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_labeled.png"))
    plt.close()

    print("\nDETAILED CLASS-WISE BREAKDOWN:")

    for i, cname in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        print(f"\n▶ Class: {cname}")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")

        if fp > 0:
            print("  Misclassified AS this class[False Positives]:")
            for j in range(NUM_CLASSES):
                if j != i and cm[j, i] > 0:
                    print(f"    {CLASS_NAMES[j]} → {cname}: {cm[j, i]}")

        if fn > 0:
            print("  Misclassified FROM this class[False Negatives]:")
            for j in range(NUM_CLASSES):
                if j != i and cm[i, j] > 0:
                    print(f"    {cname} → {CLASS_NAMES[j]}: {cm[i, j]}")

if __name__ == "__main__":
    main()

'''






















'''
# 18/12/25(below all latest)
#Final one sent to Amit sir with charts
# train_color_final_hard_guarantee.py
"""
Full training pipeline (Option A - Hard Guarantee):
- CSV preprocessing
- Color-only feature extraction (5x5 patch)
- Augmentation + oversampling
- Class-guaranteed image-level split (duplicates allowed when necessary)
- MLP + RandomForest + KNN ensemble
- RF/KNN probability expansion to full class vector
- Always prints 4-class classification_report safely

MODIFIED AS REQUESTED:
- EarlyStopping REMOVED
- ReduceLROnPlateau KEPT
- Training runs full 300 epochs
- Added 3 plots (accuracy vs epochs, train/val loss, val loss vs epochs)
- FIXED LABEL NORMALIZATION: lower-case check + proper order
"""

import os
import cv2
import random
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ---------------- CONFIG ----------------
RND = 42
random.seed(RND)
np.random.seed(RND)
tf.random.set_seed(RND)

CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "images"
MODEL_DIR = "model"

PATCH_HALF = 2
NOISY_STD_THRESH = 0.045
AUGMENT_COPIES = 2
BATCH_SIZE = 64
EPOCHS = 300  # FULL 300 EPOCHS ALWAYS

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

NUM_CLASSES = len(LABEL_MAP)
os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================================
#  FIXED LABEL NORMALIZATION  (LOWERCASE + CORRECT ORDER)
# =====================================================================
def normalize_label(s):
    """
    Correct normalization:
    - Convert to lowercase
    - Remove spaces, hyphens, underscores
    - Detect 'non-green' before 'green' to avoid false matches
    """
    s = str(s).strip().lower()

    # remove separators
    s = s.replace(" ", "").replace("_", "").replace("-", "")

    # detect NON-GREEN FIRST (critical: otherwise 'green' matches inside 'nongreen')
    if "nongreen800" in s or "nongreen" in s:
        return "non-green-800"

    # detect GREEN
    if "green800" in s:
        return "green-800"

    # detect SEA
    if "sea800" in s or s == "sea":
        return "sea-800"

    # detect COCONUT
    if "coconut800" in s or "coconut" in s:
        return "coconut-800"

    # fallback
    return None


# =====================================================================
#  CSV PREPROCESSING
# =====================================================================
def preprocess_csv(csv_path, image_folder):
    df = pd.read_csv(csv_path)

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])

    def clean_points(p):
        p = str(p).strip()
        if not p:
            return None
        pts = []
        for token in p.split(";"):
            if "," not in token:
                continue
            try:
                x, y = token.split(",")
                pts.append(f"{float(x)},{float(y)}")
            except:
                continue
        return ";".join(pts) if pts else None

    df["points"] = df["points"].apply(clean_points)
    df = df.dropna(subset=["points"])

    df = df.drop_duplicates(subset=["filename", "label", "points"])

    df["filepath"] = df["filename"].apply(lambda f: os.path.join(image_folder, str(f)))
    df = df[df["filepath"].apply(os.path.exists)]

    # require at least 2 annotated points per image
    valid_imgs = df.groupby("filename")["points"].count()
    df = df[df["filename"].isin(valid_imgs[valid_imgs >= 2].index)]

    return df.reset_index(drop=True)


# =====================================================================
#  COLOR FEATURE EXTRACTION (unchanged)
# =====================================================================
def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Missing image: {path}")
    return img

def compute_patch(img, x, y, h=PATCH_HALF):
    H, W = img.shape[:2]
    if x < h or y < h or x >= W-h or y >= H-h:
        return None
    return img[y-h:y+h+1, x-h:x+h+1]

def patch_is_clean(lab_uint8):
    p = lab_uint8.astype(np.float32) / 255.0
    return p.std(axis=(0,1)).max() < NOISY_STD_THRESH

def compute_edge_mean(l_uint8):
    sx = cv2.Sobel(l_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(l_uint8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx*sx + sy*sy)
    return float(mag.mean()) / 255.0

def rgb_ratios_from_patch(patch):
    h, w = patch.shape[:2]
    cx, cy = w//2, h//2
    b, g, r = patch[cy, cx].astype(np.float32)
    denom = r + g + b + 1e-6
    return float(b/denom), float(g/denom)

def extract_features_for_point(img_bgr, x, y):
    patch = compute_patch(img_bgr, x, y)
    if patch is None:
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ycr = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

    if not patch_is_clean(lab):
        return None

    lab_f = lab.astype(np.float32)/255.0
    hsv_f = hsv.astype(np.float32)/255.0
    ycr_f = ycr.astype(np.float32)/255.0

    return np.array([
        lab_f[...,0].mean(), lab_f[...,1].mean(), lab_f[...,2].mean(),
        lab_f[...,0].std(),  lab_f[...,1].std(),  lab_f[...,2].std(),
        hsv_f[...,0].mean(), hsv_f[...,1].mean(), hsv_f[...,2].mean(),
        hsv_f[...,0].std(),  hsv_f[...,1].std(),  hsv_f[...,2].std(),
        ycr_f[...,0].mean(), ycr_f[...,1].mean(), ycr_f[...,2].mean(),
        *rgb_ratios_from_patch(patch),
        compute_edge_mean(lab[...,0])
    ], dtype=np.float32)


# =====================================================================
#  LOAD FEATURES GROUPED PER IMAGE
# =====================================================================
def load_grouped_features(csv_file, image_folder):
    df = preprocess_csv(csv_file, image_folder)
    grouped = defaultdict(list)
    total = 0

    for _, row in df.iterrows():
        fname = row["filename"]
        label = LABEL_MAP[row["label"]]

        img = safe_imread(os.path.join(image_folder, fname))
        if img is None:
            continue

        for token in row["points"].split(";"):
            x, y = map(float, token.split(","))
            feat = extract_features_for_point(img, int(x), int(y))
            if feat is not None:
                grouped[fname].append((feat, label))
                total += 1

    print(f"\nLoaded {total} valid cleaned samples from {len(grouped)} images.")
    return grouped


# =====================================================================
#  TRAIN/TEST SPLIT + GUARANTEE (unchanged)
# =====================================================================
def initial_stratified_split(grouped, test_size=0.2):
    imgs = []
    primary = []

    for fname, items in grouped.items():
        cnt = defaultdict(int)
        for _, lbl in items:
            cnt[lbl] += 1
        primary.append(max(cnt.items(), key=lambda x: x[1])[0])
        imgs.append(fname)

    train_imgs, test_imgs = train_test_split(
        imgs, test_size=test_size, stratify=primary, random_state=RND
    )
    return list(train_imgs), list(test_imgs)


def ensure_class_guarantee(train_imgs, test_imgs, grouped, min_per_set=1):
    class_to_images = defaultdict(list)
    for fname, items in grouped.items():
        classes_in_img = {lbl for _, lbl in items}
        for c in classes_in_img:
            class_to_images[c].append(fname)

    def count_in(imgs, cls):
        return sum(1 for im in set(imgs) if cls in {lbl for _, lbl in grouped[im]})

    changed = True
    while changed:
        changed = False
        for cls in range(NUM_CLASSES):
            train_count = count_in(train_imgs, cls)
            test_count = count_in(test_imgs, cls)

            # fix train
            if train_count < min_per_set:
                if class_to_images[cls]:
                    chosen = class_to_images[cls][0]
                    if chosen not in train_imgs:
                        train_imgs.append(chosen)
                    if chosen not in test_imgs:
                        test_imgs.append(chosen)
                    changed = True
                continue

            # fix test
            if test_count < min_per_set:
                if class_to_images[cls]:
                    chosen = class_to_images[cls][0]
                    if chosen not in test_imgs:
                        test_imgs.append(chosen)
                    if chosen not in train_imgs:
                        train_imgs.append(chosen)
                    changed = True
                continue

    return train_imgs, test_imgs


# =====================================================================
#  AUGMENTATION + OVERSAMPLING (unchanged)
# =====================================================================
def jitter(f):
    return np.clip(f + np.random.normal(scale=0.01, size=f.shape), 0, 1)

def oversample(X, y):
    uniq, cnt = np.unique(y, return_counts=True)
    mx = cnt.max()
    outX, outY = [], []

    for cls in uniq:
        idx = np.where(y == cls)[0]
        for i in idx:
            outX.append(X[i]); outY.append(cls)
        if len(idx) < mx:
            extra = np.random.choice(idx, mx - len(idx), replace=True)
            for i in extra:
                outX.append(X[i]); outY.append(cls)

    outX = np.array(outX); outY = np.array(outY)
    perm = np.random.permutation(len(outX))
    return outX[perm], outY[perm]


# =====================================================================
#  MLP MODEL (unchanged)
# =====================================================================
def build_mlp(in_dim, classes):
    m = models.Sequential([
        layers.Input(shape=(in_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(classes, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m


# =====================================================================
#  PROBABILITY EXPANSION (unchanged)
# =====================================================================
def expand_proba(probs, model_classes, num_classes):
    N = probs.shape[0]
    out = np.zeros((N, num_classes))
    for i, cls in enumerate(model_classes):
        out[:, int(cls)] = probs[:, i]
    return out


# =====================================================================
#  MAIN TRAINING PIPELINE
# =====================================================================
def main():
    grouped = load_grouped_features(CSV_FILE, IMAGE_FOLDER)

    train_imgs, test_imgs = initial_stratified_split(grouped)
    train_imgs, test_imgs = ensure_class_guarantee(train_imgs, test_imgs, grouped)

    def collect(imgs):
        X, Y = [], []
        for fname in imgs:
            for feat, lbl in grouped[fname]:
                X.append(feat); Y.append(lbl)
        return np.array(X), np.array(Y)

    X_train, y_train = collect(train_imgs)
    X_test, y_test = collect(test_imgs)

    # augment
    Xa, Ya = [], []
    for i in range(len(X_train)):
        Xa.append(X_train[i]); Ya.append(y_train[i])
        for _ in range(AUGMENT_COPIES):
            Xa.append(jitter(X_train[i])); Ya.append(y_train[i])

    X_train = np.array(Xa)
    y_train = np.array(Ya)

    # oversample
    X_train, y_train = oversample(X_train, y_train)

    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_final.pkl"))

    # class weights
    uniq = np.unique(y_train)
    cw_vals = class_weight.compute_class_weight("balanced", classes=uniq, y=y_train)
    cw = {int(uniq[i]): float(cw_vals[i]) for i in range(len(uniq))}

    # callbacks
    rl = ReduceLROnPlateau(monitor="val_loss", patience=7, factor=0.5)

    # build model
    mlp = build_mlp(X_train_s.shape[1], NUM_CLASSES)

    history = mlp.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[rl],
        class_weight=cw,
        verbose=2
    )

    # =================================================================
    #  TRAINING PLOTS (3 REQUIRED CHARTS)
    # =================================================================

    # ACCURACY VS EPOCHS
    plt.figure(figsize=(7,5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "accuracy_vs_epochs.png"))
    plt.close()

    # TRAIN LOSS VS VAL LOSS
    plt.figure(figsize=(7,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "train_vs_val_loss.png"))
    plt.close()

    # VAL LOSS VS EPOCHS
    plt.figure(figsize=(7,5))
    plt.plot(history.history['val_loss'], label='Val Loss', color='red')
    plt.title("Validation Loss vs Epochs")
    plt.xlabel("Epochs"); plt.ylabel("Val Loss")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "val_loss_vs_epochs.png"))
    plt.close()

    print("\nSaved training curves in:", MODEL_DIR)

    # =================================================================
    #  RF + KNN + ENSEMBLE (unchanged)
    # =================================================================
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced")
    rf.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_s, y_train)

    mlp.save(os.path.join(MODEL_DIR, "mlp_final.keras"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_final.pkl"))
    joblib.dump(knn, os.path.join(MODEL_DIR, "knn_final.pkl"))

    # predictions
    mlp_prob = mlp.predict(X_test_s)
    knn_prob = expand_proba(knn.predict_proba(X_test_s), knn.classes_, NUM_CLASSES)
    rf_prob = expand_proba(rf.predict_proba(X_test), rf.classes_, NUM_CLASSES)

    ensemble = 0.5*mlp_prob + 0.25*rf_prob + 0.25*knn_prob
    pred = np.argmax(ensemble, axis=1)

    print("\n★ ENSEMBLE ACCURACY: {:.2f}%".format(100*accuracy_score(y_test, pred)))

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test, pred,
        labels=list(range(NUM_CLASSES)),
        target_names=list(LABEL_MAP.keys()),
        zero_division=0
    ))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, pred))

    print("\nModels saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
'''












'''
# =====================================================================
#              train_color_final_hard_guarantee_30runs.py
# =====================================================================

import os
import cv2
import random
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ---------------- CONFIG ----------------
CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "images"
MODEL_DIR = "model_30runs"   # prevent overwriting older models

PATCH_HALF = 2            # 5×5 patch
NOISY_STD_THRESH = 0.045
AUGMENT_COPIES = 2
BATCH_SIZE = 64
EPOCHS = 300

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

NUM_CLASSES = len(LABEL_MAP)

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------- Ensure reproducibility per run ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------- CSV PREPROCESSING ----------------
def preprocess_csv(csv_path, image_folder):
    df = pd.read_csv(csv_path)

    def normalize_label(s):
        s = str(s).strip().lower()
        s = s.replace(" ", "").replace("_", "").replace("-", "")
        if "green800" in s:
            return "green-800"
        if "nongreen800" in s or "nongreen" in s:
            return "non-green-800"
        if "sea800" in s or s == "sea":
            return "sea-800"
        if "coconut800" in s or "coconut" in s:
            return "coconut-800"
        return None

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])

    def clean_pts(p):
        p = str(p).strip()
        if not p:
            return None
        pts = []
        for token in p.split(";"):
            if "," not in token:
                continue
            try:
                x, y = map(float, token.split(","))
                pts.append(f"{x},{y}")
            except:
                continue
        return ";".join(pts) if pts else None

    df["points"] = df["points"].apply(clean_pts)
    df = df.dropna(subset=["points"])

    df = df.drop_duplicates(subset=["filename", "label", "points"])

    df["filepath"] = df["filename"].apply(lambda f: os.path.join(image_folder, str(f)))
    df = df[df["filepath"].apply(os.path.exists)]

    # keep images that have at least 2 annotated points
    counts = df.groupby("filename")["points"].count()
    valid = counts[counts >= 2].index
    df = df[df["filename"].isin(valid)]

    df = df.reset_index(drop=True)
    return df


# ---------------- FEATURE EXTRACTION ----------------
def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Missing: {path}")
    return img

def compute_patch(img, x, y, h=PATCH_HALF):
    H, W = img.shape[:2]
    if x < h or y < h or x >= W-h or y >= H-h:
        return None
    return img[y-h:y+h+1, x-h:x+h+1]

def patch_is_clean(lab_uint8):
    p = lab_uint8.astype(np.float32) / 255.0
    return p.std(axis=(0, 1)).max() < NOISY_STD_THRESH

def edge_mean(L):
    sx = cv2.Sobel(L, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(L, cv2.CV_32F, 0, 1)
    return float(np.sqrt(sx*sx + sy*sy).mean()) / 255.0

def rgb_ratios(patch):
    h, w = patch.shape[:2]
    cx, cy = w//2, h//2
    b, g, r = patch[cy, cx].astype(float)
    denom = r + g + b + 1e-6
    return b/denom, g/denom

def extract_features(img_bgr, x, y):
    patch = compute_patch(img_bgr, x, y)
    if patch is None:
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ycr = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

    if not patch_is_clean(lab):
        return None

    labf = lab.astype(np.float32)/255.0
    hsvf = hsv.astype(np.float32)/255.0
    ycrf = ycr.astype(np.float32)/255.0

    feat = np.array([
        labf[...,0].mean(), labf[...,1].mean(), labf[...,2].mean(),
        labf[...,0].std(),  labf[...,1].std(),  labf[...,2].std(),
        hsvf[...,0].mean(), hsvf[...,1].mean(), hsvf[...,2].mean(),
        hsvf[...,0].std(),  hsvf[...,1].std(),  hsvf[...,2].std(),
        ycrf[...,0].mean(), ycrf[...,1].mean(), ycrf[...,2].mean(),
        *rgb_ratios(patch),
        edge_mean(lab[...,0])
    ], dtype=np.float32)

    return feat


# ---------------- LOAD FEATURES ----------------
def load_grouped_features(csv_file, image_folder):
    df = preprocess_csv(csv_file, image_folder)
    grouped = defaultdict(list)
    count = 0
    for _, r in df.iterrows():
        fname = r["filename"]
        label = LABEL_MAP[r["label"]]
        img = safe_imread(os.path.join(image_folder, fname))
        if img is None:
            continue
        for token in r["points"].split(";"):
            x, y = map(float, token.split(","))
            feat = extract_features(img, int(x), int(y))
            if feat is not None:
                grouped[fname].append((feat, label))
                count += 1
    print("Loaded features:", count)
    return grouped


# ---------------- TRAIN/TEST IMAGE SPLITS ----------------
def stratified_split(grouped, test_size=0.2):
    imgs, prim = [], []
    for fname, items in grouped.items():
        cnt = defaultdict(int)
        for _, lbl in items: cnt[lbl] += 1
        primary_class = max(cnt.items(), key=lambda x:x[1])[0]
        imgs.append(fname)
        prim.append(primary_class)
    tr, te = train_test_split(imgs, test_size=test_size, stratify=prim, random_state=random.randint(1,99999))
    return tr, te


# ---------------- JITTER & OVERSAMPLING ----------------
def jitter(f):
    return np.clip(f + np.random.normal(scale=0.01, size=f.shape), 0, 1)

def oversample(X, y):
    uniq, cnt = np.unique(y, return_counts=True)
    mx = cnt.max()
    newX, newY = [], []
    for cls in uniq:
        idx = np.where(y == cls)[0]
        for i in idx:
            newX.append(X[i]); newY.append(cls)
        if len(idx) < mx:
            extra = np.random.choice(idx, mx-len(idx), replace=True)
            for i in extra:
                newX.append(X[i]); newY.append(cls)
    newX = np.array(newX); newY = np.array(newY)
    p = np.random.permutation(len(newX))
    return newX[p], newY[p]


# ---------------- MLP MODEL ----------------
def build_mlp(input_dim):
    m = models.Sequential([
        layers.Input(input_dim),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# ---------------- EXPAND PROBABILITIES ----------------
def expand_proba(probs, model_classes):
    out = np.zeros((probs.shape[0], NUM_CLASSES))
    for i, cls in enumerate(model_classes):
        out[:, int(cls)] = probs[:, i]
    return out


# ---------------- MAIN TRAINING (RUN ONCE) ----------------
def train_once(run_id=0):
    set_seed(1000 + run_id)

    grouped = load_grouped_features(CSV_FILE, IMAGE_FOLDER)

    train_imgs, test_imgs = stratified_split(grouped)

    # collect features
    def collect(imgs):
        X, Y = [], []
        for f in imgs:
            for feat, lbl in grouped[f]:
                X.append(feat); Y.append(lbl)
        return np.array(X), np.array(Y)

    X_train, y_train = collect(train_imgs)
    X_test, y_test = collect(test_imgs)

    # augment
    X_aug, y_aug = [], []
    for i in range(len(X_train)):
        X_aug.append(X_train[i]); y_aug.append(y_train[i])
        for _ in range(AUGMENT_COPIES):
            X_aug.append(jitter(X_train[i]))
            y_aug.append(y_train[i])

    X_train = np.array(X_aug)
    y_train = np.array(y_aug)

    # oversample
    X_train, y_train = oversample(X_train, y_train)

    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # class weights
    uniq = np.unique(y_train)
    cwv = class_weight.compute_class_weight("balanced", uniq, y_train)
    weights = {int(uniq[i]): float(cwv[i]) for i in range(len(uniq))}

    # MLP
    mlp = build_mlp(X_train_s.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

    mlp.fit(X_train_s, y_train, validation_data=(X_test_s, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[es, rl], class_weight=weights, verbose=0)

    # RF & KNN
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced")
    rf.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_s, y_train)

    # probabilities
    p_mlp = mlp.predict(X_test_s, verbose=0)
    p_knn = expand_proba(knn.predict_proba(X_test_s), knn.classes_)
    p_rf  = expand_proba(rf.predict_proba(X_test), rf.classes_)

    # ensemble
    p = 0.5*p_mlp + 0.25*p_rf + 0.25*p_knn
    pred = np.argmax(p, axis=1)

    acc = accuracy_score(y_test, pred)

    print(f"\n================ RUN {run_id+1} ================")
    print("Accuracy:", acc * 100)

    print("\nClassification Report:")
    print(classification_report(
        y_test, pred,
        labels=list(range(NUM_CLASSES)),
        target_names=list(LABEL_MAP.keys()),
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

    return acc


# =====================================================================
#                           RUN 30 TIMES
# =====================================================================

if __name__ == "__main__":
    accuracies = []

    for r in range(30):
        acc = train_once(r)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    mean = accuracies.mean() * 100
    std = accuracies.std() * 100

    print("\n\n================ FINAL SUMMARY ================")
    print("Accuracies:", accuracies * 100)
    print(f"\nMean Accuracy: {mean:.2f}%")
    print(f"Std Deviation: {std:.2f}%")
    print("==============================================")
'''











'''
### THE FINAL CODE

# train_color_final_hard_guarantee.py
"""
Full training pipeline (Option A - Hard Guarantee):
- CSV preprocessing
- Color-only feature extraction (5x5 patch)
- Augmentation + oversampling
- Class-guaranteed image-level split (duplicates allowed when necessary)
- MLP + RandomForest + KNN ensemble
- RF/KNN probability expansion to full class vector
- Always prints 4-class classification_report safely
"""
import os
import cv2
import random
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------- CONFIG ----------------
RND = 42
random.seed(RND)
np.random.seed(RND)
tf.random.set_seed(RND)

CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "images"
MODEL_DIR = "model"

PATCH_HALF = 2            # 5x5 patch
NOISY_STD_THRESH = 0.045
AUGMENT_COPIES = 2
BATCH_SIZE = 64
EPOCHS = 300

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

NUM_CLASSES = len(LABEL_MAP)

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------- CSV PREPROCESSING ----------------
def preprocess_csv(csv_path, image_folder):
    df = pd.read_csv(csv_path)

    def normalize_label(s):
        s = str(s).strip().lower()
        s = s.replace(" ", "").replace("_", "").replace("-", "")
        if "green800" in s:
            return "green-800"
        if "nongreen800" in s or "nongreen" in s:
            return "non-green-800"
        if "sea800" in s or "sea" == s:
            return "sea-800"
        if "coconut800" in s or "coconut" in s:
            return "coconut-800"
        return None

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])

    def clean_points(p):
        p = str(p).strip()
        if not p:
            return None
        pts = []
        for token in p.split(";"):
            if "," not in token:
                continue
            try:
                x, y = token.split(",")
                x = float(x); y = float(y)
                pts.append(f"{x},{y}")
            except:
                continue
        return ";".join(pts) if len(pts) > 0 else None

    df["points"] = df["points"].apply(clean_points)
    df = df.dropna(subset=["points"])

    df = df.drop_duplicates(subset=["filename", "label", "points"])

    df["filepath"] = df["filename"].apply(lambda f: os.path.join(image_folder, str(f)))
    df = df[df["filepath"].apply(os.path.exists)]

    counts = df.groupby("filename")["points"].count()
    valid_imgs = counts[counts >= 2].index
    df = df[df["filename"].isin(valid_imgs)]

    df = df.reset_index(drop=True)
    return df


# ---------------- color-only feature extraction ----------------
def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Missing image: {path}")
    return img

def compute_patch(img, x, y, h=PATCH_HALF):
    H, W = img.shape[:2]
    if x < h or y < h or x >= W-h or y >= H-h:
        return None
    return img[y-h:y+h+1, x-h:x+h+1]

def patch_is_clean(lab_uint8):
    p = lab_uint8.astype(np.float32) / 255.0
    return p.std(axis=(0,1)).max() < NOISY_STD_THRESH

def compute_edge_mean(l_uint8):
    sx = cv2.Sobel(l_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(l_uint8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx*sx + sy*sy)
    return float(mag.mean()) / 255.0

def rgb_ratios_from_patch(patch):
    h, w = patch.shape[:2]
    cx, cy = w//2, h//2
    b, g, r = patch[cy, cx].astype(np.float32)
    denom = r + g + b + 1e-6
    return float(b/denom), float(g/denom)

def extract_features_for_point(img_bgr, x, y):
    patch = compute_patch(img_bgr, x, y)
    if patch is None:
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ycr = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

    if not patch_is_clean(lab):
        return None

    lab_f = lab.astype(np.float32)/255.0
    hsv_f = hsv.astype(np.float32)/255.0
    ycr_f = ycr.astype(np.float32)/255.0

    feat = np.array([
        lab_f[...,0].mean(), lab_f[...,1].mean(), lab_f[...,2].mean(),
        lab_f[...,0].std(),  lab_f[...,1].std(),  lab_f[...,2].std(),
        hsv_f[...,0].mean(), hsv_f[...,1].mean(), hsv_f[...,2].mean(),
        hsv_f[...,0].std(),  hsv_f[...,1].std(),  hsv_f[...,2].std(),
        ycr_f[...,0].mean(), ycr_f[...,1].mean(), ycr_f[...,2].mean(),
        *rgb_ratios_from_patch(patch),
        compute_edge_mean(lab[...,0])
    ], dtype=np.float32)

    return feat


# ---------------- load grouped features per image ----------------
def load_grouped_features(csv_file, image_folder):
    df = preprocess_csv(csv_file, image_folder)
    grouped = defaultdict(list)
    total = 0
    for _, row in df.iterrows():
        fname = row["filename"]
        label_name = row["label"]
        if label_name not in LABEL_MAP:
            continue
        label = LABEL_MAP[label_name]
        img = safe_imread(os.path.join(image_folder, fname))
        if img is None:
            continue
        for token in row["points"].split(";"):
            x, y = map(float, token.split(","))
            feat = extract_features_for_point(img, int(x), int(y))
            if feat is not None:
                grouped[fname].append((feat, label))
                total += 1
    if total == 0:
        raise RuntimeError("No features extracted. Check CSV/images/thresholds.")
    print(f"\nLoaded {total} valid cleaned samples from {len(grouped)} images.")
    return grouped


# ---------------- initial stratified split + guarantee (Option A) ----------------
def initial_stratified_split(grouped, test_size=0.2):
    # compute primary class per image
    imgs = []
    primary = []
    for fname, items in grouped.items():
        cnt = defaultdict(int)
        for _, lbl in items:
            cnt[lbl] += 1
        primary_class = max(cnt.items(), key=lambda kv: kv[1])[0]
        imgs.append(fname)
        primary.append(primary_class)
    train_imgs, test_imgs = train_test_split(imgs, test_size=test_size, stratify=primary, random_state=RND)
    return list(train_imgs), list(test_imgs)

def ensure_class_guarantee(train_imgs, test_imgs, grouped, min_per_set=1):
    """
    Option A: hard guarantee.
    Ensures that each class appears at least `min_per_set` images in both train and test.
    If a class is missing in a set, we try to move an image from the other set.
    If the class exists in only one image total, we will DUPLICATE that image into the missing set.
    Duplication = same filename present in both lists (intentional for Option A).
    """
    all_images = list(grouped.keys())
    class_to_images = defaultdict(list)
    for fname, items in grouped.items():
        classes_in_img = set(lbl for _, lbl in items)
        for c in classes_in_img:
            class_to_images[c].append(fname)

    # helper counts
    def count_images_with_class(img_list, cls):
        return sum(1 for im in set(img_list) if cls in {lbl for _, lbl in grouped[im]})

    changed = True
    while changed:
        changed = False
        for cls in range(NUM_CLASSES):
            train_count = count_images_with_class(train_imgs, cls)
            test_count = count_images_with_class(test_imgs, cls)

            # ensure train has at least min_per_set
            if train_count < min_per_set:
                # try to find image in test_imgs that contains cls to move
                candidate = None
                for im in test_imgs:
                    if any(lbl == cls for _, lbl in grouped[im]):
                        candidate = im
                        break
                if candidate:
                    # move candidate from test to train (remove from test)
                    test_imgs = [im for im in test_imgs if im != candidate]
                    if candidate not in train_imgs:
                        train_imgs.append(candidate)
                    changed = True
                    continue
                # else no candidate in test -> find any image in dataset
                if class_to_images.get(cls):
                    chosen = class_to_images[cls][0]
                    # duplicate (add to train if not present)
                    if chosen not in train_imgs:
                        train_imgs.append(chosen)
                    # ensure also in test (duplication allowed in Option A)
                    if chosen not in test_imgs:
                        test_imgs.append(chosen)
                    changed = True
                    continue
            # ensure test has at least min_per_set
            if test_count < min_per_set:
                candidate = None
                for im in train_imgs:
                    if any(lbl == cls for _, lbl in grouped[im]):
                        candidate = im
                        break
                if candidate:
                    # move candidate from train to test (remove from train)
                    train_imgs = [im for im in train_imgs if im != candidate]
                    if candidate not in test_imgs:
                        test_imgs.append(candidate)
                    changed = True
                    continue
                if class_to_images.get(cls):
                    chosen = class_to_images[cls][0]
                    if chosen not in test_imgs:
                        test_imgs.append(chosen)
                    if chosen not in train_imgs:
                        train_imgs.append(chosen)
                    changed = True
                    continue
    # final: allow duplicates (an image may appear in both lists)
    return train_imgs, test_imgs


# ---------------- augmentation & oversample ----------------
def jitter(f):
    f = f.copy()
    f += np.random.normal(scale=0.01, size=f.shape)
    return np.clip(f, 0, 1)

def oversample(X, y):
    uniq, cnt = np.unique(y, return_counts=True)
    mx = cnt.max()
    newX, newY = [], []
    for cls in uniq:
        idx = np.where(y == cls)[0]
        for i in idx:
            newX.append(X[i]); newY.append(cls)
        if len(idx) < mx:
            extra = np.random.choice(idx, size=mx - len(idx), replace=True)
            for i in extra:
                newX.append(X[i]); newY.append(cls)
    newX = np.array(newX); newY = np.array(newY)
    perm = np.random.permutation(len(newX))
    return newX[perm], newY[perm]


# ---------------- build MLP ----------------
def build_mlp(in_dim, classes):
    m = models.Sequential([
        layers.Input(shape=(in_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(classes, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


# ---------------- expand proba helper ----------------
def expand_proba(probs, model_classes, num_classes):
    N = probs.shape[0]
    out = np.zeros((N, num_classes), dtype=np.float32)
    for i, cls in enumerate(model_classes):
        if 0 <= int(cls) < num_classes:
            out[:, int(cls)] = probs[:, i]
    return out


# ---------------- training pipeline ----------------
def main():
    grouped = load_grouped_features(CSV_FILE, IMAGE_FOLDER)

    # initial stratified split
    train_imgs, test_imgs = initial_stratified_split(grouped, test_size=0.2)
    print(f"Initial split: train images={len(train_imgs)}, test images={len(test_imgs)}")

    # ensure both sets have at least one image per class (Option A: duplicates allowed)
    train_imgs, test_imgs = ensure_class_guarantee(train_imgs, test_imgs, grouped, min_per_set=1)
    print(f"After guarantee: train images={len(train_imgs)}, test images={len(test_imgs)}")

    # build datasets (note: images may be duplicated across sets in Option A)
    def collect(img_list):
        feats, labels = [], []
        for fname in img_list:
            for feat, lbl in grouped[fname]:
                feats.append(feat); labels.append(lbl)
        return np.array(feats), np.array(labels)

    X_train, y_train = collect(train_imgs)
    X_test, y_test = collect(test_imgs)
    print(f"Collected training samples: {X_train.shape}, test samples: {X_test.shape}")

    # augment
    X_aug, y_aug = [], []
    for i in range(len(X_train)):
        X_aug.append(X_train[i]); y_aug.append(y_train[i])
        for _ in range(AUGMENT_COPIES):
            X_aug.append(jitter(X_train[i])); y_aug.append(y_train[i])
    X_train = np.array(X_aug); y_train = np.array(y_aug)

    # oversample to balance classes
    X_train, y_train = oversample(X_train, y_train)
    print("After oversampling:", X_train.shape)

    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_final.pkl"))

    # class weights
    uniq = np.unique(y_train)
    cwv = class_weight.compute_class_weight("balanced", classes=uniq, y=y_train)
    cw = {int(uniq[i]): float(cwv[i]) for i in range(len(uniq))}
    print("Class weights:", cw)

    # MLP
    mlp = build_mlp(X_train_s.shape[1], NUM_CLASSES)
    es = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5)

    mlp.fit(X_train_s, y_train, validation_data=(X_test_s, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, rl],
            class_weight=cw, verbose=2)

    # RF and KNN training
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RND)
    rf.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_s, y_train)

    # save models
    mlp.save(os.path.join(MODEL_DIR, "mlp_final.keras"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_final.pkl"))
    joblib.dump(knn, os.path.join(MODEL_DIR, "knn_final.pkl"))

    # predict probabilities
    mlp_prob = mlp.predict(X_test_s)

    knn_prob_raw = knn.predict_proba(X_test_s)
    knn_classes = knn.classes_
    knn_prob = expand_proba(knn_prob_raw, knn_classes, NUM_CLASSES)
    if knn_prob_raw.shape[1] != NUM_CLASSES:
        print(f"[WARN] KNN had {knn_prob_raw.shape[1]} classes; expanded to {NUM_CLASSES}.")

    rf_prob_raw = rf.predict_proba(X_test)
    rf_classes = rf.classes_
    rf_prob = expand_proba(rf_prob_raw, rf_classes, NUM_CLASSES)
    if rf_prob_raw.shape[1] != NUM_CLASSES:
        print(f"[WARN] RF had {rf_prob_raw.shape[1]} classes; expanded to {NUM_CLASSES}.")

    # ensemble
    ensemble = (0.5 * mlp_prob + 0.25 * rf_prob + 0.25 * knn_prob)
    pred = np.argmax(ensemble, axis=1)

    print("\n★ ENSEMBLE ACCURACY: {:.2f}%".format(accuracy_score(y_test, pred) * 100))

    # safe classification report (always print all labels)
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, pred,
          labels=list(range(NUM_CLASSES)),
          target_names=list(LABEL_MAP.keys()),
          zero_division=0))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, pred))

    print("\nModels saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
'''
















'''
//92.4 with error

import os
import cv2
import math
import random
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ---------------- CONFIG ----------------
RND = 42
random.seed(RND)
np.random.seed(RND)
tf.random.set_seed(RND)

CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "images"
MODEL_DIR = "model"

PATCH_HALF = 2            # 5x5 patch
NOISY_STD_THRESH = 0.045
AUGMENT_COPIES = 2
BATCH_SIZE = 64
EPOCHS = 300

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

os.makedirs(MODEL_DIR, exist_ok=True)



# ============================================================
#                 CSV PREPROCESSING
# ============================================================
def preprocess_csv(csv_path, image_folder):
    df = pd.read_csv(csv_path)

    def normalize_label(s):
        s = str(s).strip().lower()
        s = s.replace(" ", "").replace("_", "").replace("-", "")
        if "green800" in s:
            return "green-800"
        if "nongreen800" in s:
            return "non-green-800"
        if "sea800" in s:
            return "sea-800"
        if "coconut800" in s:
            return "coconut-800"
        return None

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])

    def clean_points(p):
        p = str(p).strip()
        if not p:
            return None
        pts = []
        for token in p.split(";"):
            if "," not in token:
                continue
            try:
                x, y = token.split(",")
                x = float(x)
                y = float(y)
                pts.append(f"{x},{y}")
            except:
                continue
        return ";".join(pts) if len(pts) > 0 else None

    df["points"] = df["points"].apply(clean_points)
    df = df.dropna(subset=["points"])

    df = df.drop_duplicates(subset=["filename", "label", "points"])

    df["filepath"] = df["filename"].apply(lambda f: os.path.join(image_folder, f))
    df = df[df["filepath"].apply(os.path.exists)]

    counts = df.groupby("filename")["points"].count()
    valid_imgs = counts[counts >= 2].index
    df = df[df["filename"].isin(valid_imgs)]

    df = df.reset_index(drop=True)
    return df



# ============================================================
#             COLOR-ONLY FEATURE EXTRACTION
# ============================================================
def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Missing image: {path}")
    return img

def compute_patch(img, x, y, h=PATCH_HALF):
    H, W = img.shape[:2]
    if x < h or y < h or x >= W-h or y >= H-h:
        return None
    return img[y-h:y+h+1, x-h:x+h+1]

def patch_is_clean(lab_uint8):
    p = lab_uint8.astype(np.float32) / 255.0
    return p.std(axis=(0,1)).max() < NOISY_STD_THRESH

def compute_edge_mean(l_uint8):
    sx = cv2.Sobel(l_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(l_uint8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx*sx + sy*sy)
    return float(mag.mean()) / 255.0

def rgb_ratios_from_patch(patch):
    h, w = patch.shape[:2]
    cx, cy = w//2, h//2
    b, g, r = patch[cy, cx].astype(np.float32)
    denom = r + g + b + 1e-6
    return float(b/denom), float(g/denom)

def extract_features_for_point(img_bgr, x, y):
    patch = compute_patch(img_bgr, x, y)
    if patch is None:
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ycr = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

    if not patch_is_clean(lab):
        return None

    lab_f = lab.astype(np.float32)/255.0
    hsv_f = hsv.astype(np.float32)/255.0
    ycr_f = ycr.astype(np.float32)/255.0

    feat = np.array([
        # LAB mean + std
        lab_f[...,0].mean(), lab_f[...,1].mean(), lab_f[...,2].mean(),
        lab_f[...,0].std(),  lab_f[...,1].std(),  lab_f[...,2].std(),

        # HSV mean + std
        hsv_f[...,0].mean(), hsv_f[...,1].mean(), hsv_f[...,2].mean(),
        hsv_f[...,0].std(),  hsv_f[...,1].std(),  hsv_f[...,2].std(),

        # YCrCb mean
        ycr_f[...,0].mean(), ycr_f[...,1].mean(), ycr_f[...,2].mean(),

        # Ratios + smoothness
        *rgb_ratios_from_patch(patch),
        compute_edge_mean(lab[...,0])

    ], dtype=np.float32)

    return feat



# ============================================================
#              LOAD GROUPED FEATURES
# ============================================================
def load_grouped_features(csv_file, image_folder):
    df = preprocess_csv(csv_file, image_folder)

    grouped = defaultdict(list)
    total = 0

    for _, row in df.iterrows():
        fname = row["filename"]
        label = LABEL_MAP[row["label"]]

        img = safe_imread(os.path.join(image_folder, fname))
        if img is None:
            continue

        for token in row["points"].split(";"):
            x, y = map(float, token.split(","))
            feat = extract_features_for_point(img, int(x), int(y))
            if feat is not None:
                grouped[fname].append((feat, label))
                total += 1

    print(f"\nLoaded {total} valid cleaned samples from {len(grouped)} images.")
    return grouped



# ============================================================
#             STRATIFIED IMAGE SPLITTING
# ============================================================
def stratified_image_split(grouped, test_size=0.2):
    ims = []
    prim = []

    for fname, items in grouped.items():
        cnt = defaultdict(int)
        for _, lbl in items:
            cnt[lbl] += 1
        primary = max(cnt.items(), key=lambda x:x[1])[0]
        ims.append(fname)
        prim.append(primary)

    tr, te = train_test_split(
        ims,
        test_size=test_size,
        stratify=prim,
        random_state=RND
    )
    return tr, te



# ============================================================
#             AUGMENTATION + OVERSAMPLING
# ============================================================
def jitter(f):
    f = f.copy()
    f += np.random.normal(scale=0.01, size=f.shape)
    return np.clip(f, 0, 1)

def oversample(X, y):
    uniq, cnt = np.unique(y, return_counts=True)
    mx = cnt.max()

    newX, newY = [], []
    for cls in uniq:
        idx = np.where(y == cls)[0]

        for i in idx:
            newX.append(X[i])
            newY.append(cls)

        if len(idx) < mx:
            extra = np.random.choice(idx, size=mx - len(idx), replace=True)
            for i in extra:
                newX.append(X[i])
                newY.append(cls)

    newX = np.array(newX)
    newY = np.array(newY)
    perm = np.random.permutation(len(newX))
    return newX[perm], newY[perm]



# ============================================================
#                           MLP
# ============================================================
def build_mlp(in_dim, classes):
    m = models.Sequential([
        layers.Input(shape=(in_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(classes, activation='softmax')
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return m



# ============================================================
# Helper to expand any model proba to full class count
# ============================================================
def expand_proba(probs, model_classes, num_classes):
    """
    probs: (N, C_model)
    model_classes: array-like of class indices that correspond to the columns in probs
    returns: (N, num_classes) where missing columns are zeros
    """
    N = probs.shape[0]
    out = np.zeros((N, num_classes), dtype=np.float32)
    for i, cls in enumerate(model_classes):
        if cls < 0 or cls >= num_classes:
            # unexpected, skip
            continue
        out[:, int(cls)] = probs[:, i]
    return out



# ============================================================
#                     TRAINING PIPELINE
# ============================================================
def main():

    grouped = load_grouped_features(CSV_FILE, IMAGE_FOLDER)
    train_imgs, test_imgs = stratified_image_split(grouped)

    def collect(files):
        F, L = [], []
        for f in files:
            for feat, lbl in grouped[f]:
                F.append(feat)
                L.append(lbl)
        return np.array(F), np.array(L)

    X_train, y_train = collect(train_imgs)
    X_test, y_test = collect(test_imgs)

    # -------- AUGMENT --------
    augX, augY = [], []
    for i in range(len(X_train)):
        augX.append(X_train[i]); augY.append(y_train[i])
        for _ in range(AUGMENT_COPIES):
            augX.append(jitter(X_train[i]))
            augY.append(y_train[i])

    X_train = np.array(augX)
    y_train = np.array(augY)

    # -------- OVERSAMPLE --------
    X_train, y_train = oversample(X_train, y_train)

    # -------- SCALE --------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_final.pkl")

    # -------- CLASS WEIGHTS --------
    uniq = np.unique(y_train)
    cwv = class_weight.compute_class_weight("balanced", classes=uniq, y=y_train)
    cw = {int(uniq[i]): float(cwv[i]) for i in range(len(uniq))}

    # -------- TRAIN MLP --------
    mlp = build_mlp(X_train_s.shape[1], len(LABEL_MAP))
    es = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5)

    mlp.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[es, rl],
        class_weight=cw,
        verbose=2
    )

    # -------- TRAIN RF + KNN --------
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RND)
    rf.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_s, y_train)

    # -------- SAVE MODELS --------
    mlp.save(f"{MODEL_DIR}/mlp_final.keras")
    joblib.dump(rf, f"{MODEL_DIR}/rf_final.pkl")
    joblib.dump(knn, f"{MODEL_DIR}/knn_final.pkl")

    # ======================================================
    #          GET PROBABILITIES & EXPAND TO FULL SHAPE
    # ======================================================
    mlp_prob = mlp.predict(X_test_s)

    # KNN probabilities (may have fewer classes)
    knn_prob_raw = knn.predict_proba(X_test_s)
    knn_classes = knn.classes_
    knn_prob = expand_proba(knn_prob_raw, knn_classes, num_classes=len(LABEL_MAP))
    if knn_prob_raw.shape[1] != len(LABEL_MAP):
        print(f"[WARN] KNN had {knn_prob_raw.shape[1]} classes; expanded to {len(LABEL_MAP)}.")

    # RF probabilities (may have fewer classes)
    rf_prob_raw = rf.predict_proba(X_test)
    rf_classes = rf.classes_
    rf_prob = expand_proba(rf_prob_raw, rf_classes, num_classes=len(LABEL_MAP))
    if rf_prob_raw.shape[1] != len(LABEL_MAP):
        print(f"[WARN] RF had {rf_prob_raw.shape[1]} classes; expanded to {len(LABEL_MAP)}.")

    # ======================================================
    #                 FINAL ENSEMBLE
    # ======================================================
    ensemble = (0.5 * mlp_prob + 0.25 * rf_prob + 0.25 * knn_prob)
    pred = np.argmax(ensemble, axis=1)

    print("\n★ ENSEMBLE ACCURACY:", accuracy_score(y_test, pred) * 100, "%")

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, pred, target_names=list(LABEL_MAP.keys())))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, pred))

    print("\nModels saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()


'''


'''
//accuracy 90%+ but code has an error

import os
import cv2
import math
import random
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------- CONFIG ----------------
RND = 42
random.seed(RND)
np.random.seed(RND)
tf.random.set_seed(RND)

CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "images"
MODEL_DIR = "model"

PATCH_HALF = 2                # 5x5 patch
NOISY_STD_THRESH = 0.045
AUGMENT_COPIES = 2
BATCH_SIZE = 64
EPOCHS = 300

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------
#                 CSV PREPROCESSING FUNCTION
# -----------------------------------------------------------
def preprocess_csv(csv_path, image_folder):
    df = pd.read_csv(csv_path)

    # --- 1. Normalize labels ---
    def normalize_label(s):
        s = str(s).strip().lower()
        s = s.replace(" ", "").replace("_", "").replace("-", "")
        if "green800" in s:
            return "green-800"
        if "nongreen800" in s:
            return "non-green-800"
        if "sea800" in s:
            return "sea-800"
        if "coconut800" in s:
            return "coconut-800"
        return None

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])

    # --- 2. Clean points ---
    def clean_points(p):
        p = str(p).strip()
        if not p:
            return None
        parts = p.split(";")
        valid = []
        for pt in parts:
            if "," not in pt:
                continue
            try:
                x, y = pt.split(",")
                x = float(x)
                y = float(y)
                valid.append(f"{x},{y}")
            except:
                continue
        return ";".join(valid) if len(valid) > 0 else None

    df["points"] = df["points"].apply(clean_points)
    df = df.dropna(subset=["points"])

    # --- 3. Remove duplicates ---
    df = df.drop_duplicates(subset=["filename", "label", "points"])

    # --- 4. Remove missing image files ---
    df["filepath"] = df["filename"].apply(lambda f: os.path.join(image_folder, str(f)))
    df = df[df["filepath"].apply(os.path.exists)]

    # --- 5. Ensure at least 2 points per image ---
    counts = df.groupby("filename")["points"].count()
    valid_images = counts[counts >= 2].index
    df = df[df["filename"].isin(valid_images)]

    df = df.reset_index(drop=True)
    return df

# -----------------------------------------------------------
#                   COLOR-ONLY FEATURE EXTRACTION
# -----------------------------------------------------------
def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Missing image: {path}")
    return img

def compute_patch(img, x, y, h=PATCH_HALF):
    H, W = img.shape[:2]
    if x < h or y < h or x >= W-h or y >= H-h:
        return None
    return img[y-h:y+h+1, x-h:x+h+1]

def patch_is_clean(lab_uint8):
    p = lab_uint8.astype(np.float32)/255.0
    return p.std(axis=(0,1)).max() < NOISY_STD_THRESH

def compute_edge_mean(l_channel_uint8):
    l = l_channel_uint8
    sx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx*sx + sy*sy)
    return float(mag.mean()) / 255.0

def rgb_ratios_from_patch(patch):
    h, w = patch.shape[:2]
    cx, cy = w//2, h//2
    b, g, r = patch[cy, cx].astype(np.float32)
    denom = r + g + b + 1e-6
    return float(b/denom), float(g/denom)

def extract_features_for_point(img_bgr, x, y):
    patch = compute_patch(img_bgr, x, y)
    if patch is None:
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ycr = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

    if not patch_is_clean(lab):
        return None

    lab_f = lab.astype(np.float32)/255.0
    hsv_f = hsv.astype(np.float32)/255.0
    ycr_f = ycr.astype(np.float32)/255.0

    lab_mean = lab_f[...,0].mean(), lab_f[...,1].mean(), lab_f[...,2].mean()
    lab_std  = lab_f[...,0].std(),  lab_f[...,1].std(),  lab_f[...,2].std()

    hsv_mean = hsv_f.mean(axis=(0,1))
    hsv_std  = hsv_f.std(axis=(0,1))

    ycr_mean = ycr_f.mean(axis=(0,1))

    blue_ratio, green_ratio = rgb_ratios_from_patch(patch)
    edge_mean = compute_edge_mean(lab[...,0])

    feat = np.array([
        lab_mean[0], lab_mean[1], lab_mean[2],
        lab_std[0],  lab_std[1],  lab_std[2],
        hsv_mean[0], hsv_mean[1], hsv_mean[2],
        hsv_std[0],  hsv_std[1],  hsv_std[2],
        ycr_mean[0], ycr_mean[1], ycr_mean[2],
        blue_ratio, green_ratio,
        edge_mean
    ], dtype=np.float32)

    return feat

# -----------------------------------------------------------
#           LOAD & GROUP FEATURES PER IMAGE (cleaned CSV)
# -----------------------------------------------------------
def load_grouped_features(csv_file, image_folder):
    df = preprocess_csv(csv_file, image_folder)

    grouped = defaultdict(list)
    total = 0

    for _, row in df.iterrows():
        fname = row["filename"]
        label = LABEL_MAP[row["label"]]
        img = safe_imread(os.path.join(image_folder, fname))
        if img is None:
            continue

        for p in row["points"].split(";"):
            if "," not in p:
                continue
            x,y = map(float, p.split(","))
            x,y = int(x), int(y)
            feat = extract_features_for_point(img, x, y)
            if feat is None:
                continue
            grouped[fname].append((feat, label))
            total += 1

    print(f"Loaded {total} cleaned color samples from {len(grouped)} images")
    return grouped

# -----------------------------------------------------------
#               STRATIFIED IMAGE SPLITTING
# -----------------------------------------------------------
def stratified_image_split(grouped, test_size=0.2):
    imgs = []
    prim = []

    for fname, items in grouped.items():
        counts = defaultdict(int)
        for _, lbl in items:
            counts[lbl]+=1
        primary = max(counts.items(), key=lambda x:x[1])[0]
        imgs.append(fname)
        prim.append(primary)

    t_train, t_test = train_test_split(
        imgs, test_size=test_size,
        stratify=prim, random_state=RND
    )
    return t_train, t_test

# -----------------------------------------------------------
#             OVERSAMPLING + AUGMENTATION
# -----------------------------------------------------------
def oversample(X,y):
    uniq, cnt = np.unique(y, return_counts=True)
    mx = cnt.max()

    newX=[]
    newy=[]

    for cls in uniq:
        idx = np.where(y==cls)[0]
        for i in idx:
            newX.append(X[i]); newy.append(cls)

        if len(idx)<mx:
            extra=np.random.choice(idx,size=mx-len(idx),replace=True)
            for i in extra:
                newX.append(X[i]); newy.append(cls)

    newX=np.stack(newX,axis=0)
    newy=np.array(newy)
    sh=np.random.permutation(len(newX))
    return newX[sh], newy[sh]

def jitter(f):
    f=f.copy()
    f+=np.random.normal(scale=0.01,size=f.shape)
    return np.clip(f,0,1)

# -----------------------------------------------------------
#                     MLP MODEL
# -----------------------------------------------------------
def build_mlp(in_dim, classes):
    m=models.Sequential([
        layers.Input(shape=(in_dim,)),
        layers.Dense(256,activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128,activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64,activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(classes,activation="softmax"),
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m

# ===========================================================
#                   TRAINING PIPELINE
# ===========================================================
def main():
    grouped = load_grouped_features(CSV_FILE, IMAGE_FOLDER)

    train_imgs, test_imgs = stratified_image_split(grouped, 0.2)

    def build_dataset(img_list):
        feats=[]
        labels=[]
        for fname in img_list:
            for f,l in grouped[fname]:
                feats.append(f); labels.append(l)
        feats=np.array(feats); labels=np.array(labels)
        return feats, labels

    X_train, y_train = build_dataset(train_imgs)
    X_test, y_test = build_dataset(test_imgs)

    # --- Augmentation ---
    augX=[]
    augY=[]
    for i in range(len(X_train)):
        augX.append(X_train[i]); augY.append(y_train[i])
        for _ in range(AUGMENT_COPIES):
            augX.append(jitter(X_train[i])); augY.append(y_train[i])

    X_train=np.array(augX)
    y_train=np.array(augY)

    # --- Oversample ---
    X_train, y_train = oversample(X_train, y_train)

    # --- Scale ---
    scaler=StandardScaler()
    X_train_s=scaler.fit_transform(X_train)
    X_test_s=scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_final.pkl")

    # --- Class weights ---
    uniq=np.unique(y_train)
    cwv=class_weight.compute_class_weight("balanced", classes=uniq, y=y_train)
    cw={int(uniq[i]):float(cwv[i]) for i in range(len(uniq))}

    # --- Train models ---
    mlp = build_mlp(X_train_s.shape[1], len(LABEL_MAP))
    es = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor="val_loss", patience=7, factor=0.5)

    mlp.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[es,rl],
        class_weight=cw,
        verbose=2
    )

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RND)
    rf.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_train_s, y_train)

    joblib.dump(rf, f"{MODEL_DIR}/rf_final.pkl")
    joblib.dump(knn, f"{MODEL_DIR}/knn_final.pkl")
    mlp.save(f"{MODEL_DIR}/mlp_final.h5")

    # --- Evaluate ---
    mlp_prob = mlp.predict(X_test_s)
    rf_prob = rf.predict_proba(X_test)
    knn_prob = knn.predict_proba(X_test_s)

    ens = (0.5*mlp_prob + 0.25*rf_prob + 0.25*knn_prob)
    pred = np.argmax(ens, axis=1)

    print("\nEnsemble Accuracy:", accuracy_score(y_test, pred)*100)
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=list(LABEL_MAP.keys())))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

if __name__=="__main__":
    main()

'''



'''
//The problem identifier code:
import os
import cv2
import math
import random
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

RND = 42
random.seed(RND)
np.random.seed(RND)
tf.random.set_seed(RND)

# ---------------- CONFIG ----------------
CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "images"
MODEL_DIR = "model"

PATCH_HALF = 1  # 3x3 neighborhood (use 1). You can increase to 2 for 5x5.
NOISY_STD_THRESH = 0.035  # threshold to reject mixed/noisy patches (tuneable)
AUGMENT_COPIES = 2  # how many augmented copies per original sample when augmenting
BATCH_SIZE = 64
EPOCHS = 300

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Could not read image: {path}")
    return img

def compute_patch(img, x, y, half=PATCH_HALF):
    h, w = img.shape[:2]
    if x < half or y < half or x >= w - half or y >= h - half:
        return None
    return img[y-half:y+half+1, x-half:x+half+1]

def patch_is_pure_lab(lab_patch, std_thresh=NOISY_STD_THRESH):
    # lab_patch expected in uint8 0..255, compute normalized std
    p = lab_patch.astype(np.float32) / 255.0
    return p.std(axis=(0,1)).max() < std_thresh

def extract_features_for_point(img_bgr, x, y):
    """
    Returns feature vector for a pixel using local neighborhood.
    Features: lab_mean(3), lab_std(3), hsv_mean(H,S), hsv_std(H,S), ycrcb_mean(3)
    Total dims = 3+3+2+2+3 = 13
    """
    patch_bgr = compute_patch(img_bgr, x, y)
    if patch_bgr is None:
        return None

    # Convert spaces
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    ycrcb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0

    # purity check on lab patch
    if not patch_is_pure_lab((lab * 255.0).astype(np.uint8)):
        return None

    lab_mean = lab[..., 0].mean(), lab[..., 1].mean(), lab[..., 2].mean()
    lab_std  = lab[..., 0].std(),  lab[..., 1].std(),  lab[..., 2].std()

    hsv_h_mean = hsv[..., 0].mean()
    hsv_s_mean = hsv[..., 1].mean()
    hsv_h_std  = hsv[..., 0].std()
    hsv_s_std  = hsv[..., 1].std()

    ycrcb_mean = ycrcb[..., 0].mean(), ycrcb[..., 1].mean(), ycrcb[..., 2].mean()

    feat = np.array([
        lab_mean[0], lab_mean[1], lab_mean[2],
        lab_std[0],  lab_std[1],  lab_std[2],
        hsv_h_mean, hsv_s_mean,
        hsv_h_std,  hsv_s_std,
        ycrcb_mean[0], ycrcb_mean[1], ycrcb_mean[2],
    ], dtype=np.float32)

    return feat

# ---------------- Load data (image-aware) ----------------
def load_all_features(csv_file, image_folder):
    df = pd.read_csv(csv_file)
    grouped = defaultdict(list)  # filename -> list of (feat, label)
    total = 0
    for _, row in df.iterrows():
        label_name = str(row['label']).strip().lower()
        if label_name not in LABEL_MAP:
            continue
        filename = row['filename']
        img_path = os.path.join(image_folder, filename)
        img = safe_imread(img_path)
        if img is None:
            continue
        points = str(row['points']).split(';')
        for pstr in points:
            if not pstr.strip() or "," not in pstr:
                continue
            try:
                x_f, y_f = map(float, pstr.strip().split(','))
            except Exception:
                continue
            x, ycoord = int(x_f), int(y_f)
            feat = extract_features_for_point(img, x, ycoord)
            if feat is None:
                continue
            grouped[filename].append((feat, LABEL_MAP[label_name]))
            total += 1
    if total == 0:
        raise RuntimeError("No valid patches/features extracted. Check CSV/images/thresholds.")
    print(f"Loaded total {total} features from {len(grouped)} images")
    return grouped

# ---------------- Oversample ----------------
def oversample_balance(X, y):
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    newX = []
    newy = []
    for cls in unique:
        idxs = np.where(y == cls)[0]
        cur = len(idxs)
        # keep original
        for i in idxs:
            newX.append(X[i])
            newy.append(cls)
        if cur < max_count:
            # sample with replacement
            choices = np.random.choice(idxs, size=(max_count - cur), replace=True)
            for i in choices:
                newX.append(X[i])
                newy.append(cls)
    newX = np.stack(newX, axis=0)
    newy = np.array(newy, dtype=np.int32)
    perm = np.random.permutation(len(newX))
    return newX[perm], newy[perm]

# ---------------- Augmentation (feature-level jitter) ----------------
def augment_feature_vector(feat, n_copies=1):
    """
    Small realistic jitter to color features:
    - add gaussian noise to lab_mean and ycrcb_mean
    - small shift to hsv_h_mean and hsv_s_mean
    - small increase/decrease to lab_std and hsv stds
    """
    copies = []
    for _ in range(n_copies):
        f = feat.copy()
        # lab mean indices 0..2
        f[0:3] += np.random.normal(scale=0.01, size=3)  # small shift
        # lab std indices 3..5
        f[3:6] = np.clip(f[3:6] + np.random.normal(scale=0.005, size=3), 0, 1)
        # hsv mean H,S indices 6,7
        f[6] = (f[6] + np.random.normal(scale=0.02)) % 1.0
        f[7] = np.clip(f[7] + np.random.normal(scale=0.02), 0, 1)
        # hsv std 8,9
        f[8:10] = np.clip(f[8:10] + np.random.normal(scale=0.005, size=2), 0, 1)
        # ycrcb mean 10..12
        f[10:13] += np.random.normal(scale=0.01, size=3)
        copies.append(f)
    return copies

# ---------------- MLP build ----------------
def build_mlp(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------- Training pipeline ----------------
def main():
    grouped = load_all_features(CSV_FILE, IMAGE_FOLDER)
    filenames = list(grouped.keys())
    # image-aware split
    train_files, test_files = train_test_split(filenames, test_size=0.2, random_state=RND)
    print(f"Train images: {len(train_files)}, Test images: {len(test_files)}")

    def gather(file_list):
        feats = []
        labels = []
        for fn in file_list:
            for feat, lbl in grouped[fn]:
                feats.append(feat)
                labels.append(lbl)
        if len(feats) == 0:
            return np.zeros((0,13)), np.zeros((0,), dtype=np.int32)
        return np.stack(feats, axis=0), np.array(labels, dtype=np.int32)

    X_train, y_train = gather(train_files)
    X_test, y_test = gather(test_files)
    print(f"Initial train patches: {X_train.shape}, test patches: {X_test.shape}")

    # Data augmentation (feature-level) on training set
    print("Applying feature-level augmentation...")
    X_aug = []
    y_aug = []
    for i in range(len(X_train)):
        X_aug.append(X_train[i])
        y_aug.append(y_train[i])
        # create small augmented copies
        aug_copies = augment_feature_vector(X_train[i], n_copies=AUGMENT_COPIES)
        for ac in aug_copies:
            X_aug.append(ac)
            y_aug.append(y_train[i])
    X_train = np.stack(X_aug, axis=0)
    y_train = np.array(y_aug, dtype=np.int32)
    print(f"After augmentation train patches: {X_train.shape}")

    # Oversample to balance classes
    print("Oversampling to balance classes...")
    X_train_bal, y_train_bal = oversample_balance(X_train, y_train)
    print(f"Balanced train patches: {X_train_bal.shape}")

    # Standardize (fit on training balanced)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "color_scaler_full.pkl"))
    print("Saved scaler.")

    # Compute class weights for MLP training (optional)
    unique = np.unique(y_train_bal)
    cw_vals = class_weight.compute_class_weight('balanced', classes=unique, y=y_train_bal)
    class_weights = {int(unique[i]): float(cw_vals[i]) for i in range(len(unique))}
    print("Class weights:", class_weights)

    # Train MLP
    input_dim = X_train_scaled.shape[1]
    mlp = build_mlp(input_dim=input_dim, num_classes=len(LABEL_MAP))
    mlp.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)

    print("Training MLP...")
    mlp.fit(X_train_scaled, y_train_bal,
            validation_data=(X_test_scaled, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weights,
            verbose=2)

    # Train RandomForest and KNN on the SAME balanced (non-scaled for RF/KNN scale is fine but we will use scaled)
    print("Training RandomForest and KNN...")
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RND)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

    rf.fit(X_train_bal, y_train_bal)
    knn.fit(X_train_scaled, y_train_bal)  # KNN on scaled features

    # Save models
    mlp.save(os.path.join(MODEL_DIR, "mlp_color_full.h5"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_color_full.pkl"))
    joblib.dump(knn, os.path.join(MODEL_DIR, "knn_color_full.pkl"))
    print("Saved MLP, RF, KNN models.")

    # Ensemble prediction on test set: average probabilities
    print("Evaluating ensemble on test set...")
    # MLP probabilities:
    mlp_probs = mlp.predict(X_test_scaled)  # shape (N, C)
    # RF probabilities (on unscaled X_test: rf trained on unscaled X_train_bal)
    # But our rf was trained on X_train_bal (unscaled). We used scaled for RF? we trained rf on X_train_bal (unscaled).
    # So prepare X_test_unscaled accordingly:
    X_test_unscaled = X_test  # X_test is unscaled original
    try:
        rf_probs = rf.predict_proba(X_test_unscaled)
    except Exception:
        # if rf expects scaled input, fallback to scaled
        rf_probs = rf.predict_proba(X_test_scaled)

    knn_probs = knn.predict_proba(X_test_scaled)

    # average probs (give mlp a little more weight)
    ensemble_probs = (0.5 * mlp_probs) + (0.25 * rf_probs) + (0.25 * knn_probs)
    y_pred_ens = np.argmax(ensemble_probs, axis=1)

    acc_ens = accuracy_score(y_test, y_pred_ens)
    print(f"\nEnsemble Accuracy: {acc_ens*100:.2f}%")

    # Print detailed metrics
    print("\nClassification Report (ensemble):")
    print(classification_report(y_test, y_pred_ens, target_names=list(LABEL_MAP.keys())))

    print("Confusion Matrix (ensemble):")
    print(confusion_matrix(y_test, y_pred_ens))

    # Also show individual model accuracies
    mlp_preds = np.argmax(mlp_probs, axis=1)
    rf_preds = np.argmax(rf_probs, axis=1)
    knn_preds = np.argmax(knn_probs, axis=1)
    print(f"MLP acc: {accuracy_score(y_test, mlp_preds)*100:.2f}%")
    print(f"RF  acc: {accuracy_score(y_test, rf_preds)*100:.2f}%")
    print(f"KNN acc: {accuracy_score(y_test, knn_preds)*100:.2f}%")

    print("All artifacts saved in", MODEL_DIR)


if __name__ == "__main__":
    main()


'''

'''
//mlp with preprocessing - 83.69%
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "images"

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

def extract_color_features(csv_file, image_folder):
    df = pd.read_csv(csv_file)

    X = []
    y = []

    for _, row in df.iterrows():
        label = row['label'].strip().lower()
        if label not in LABEL_MAP:
            continue

        filename = row['filename']
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert to LAB + HSV
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        points = str(row['points']).split(";")
        for p in points:
            if "," not in p:
                continue

            x, ycoord = map(float, p.split(","))
            x, ycoord = int(x), int(ycoord)

            if not (0 <= x < img.shape[1] and 0 <= ycoord < img.shape[0]):
                continue

            pixel_lab = lab[ycoord, x] / 255.0
            pixel_hsv = hsv[ycoord, x] / 255.0

            # Feature vector: LAB + H + S = 5D
            feat = np.array([
                pixel_lab[0], pixel_lab[1], pixel_lab[2],
                pixel_hsv[0], pixel_hsv[1]
            ])

            X.append(feat)
            y.append(LABEL_MAP[label])

    return np.array(X), np.array(y)


def build_color_mlp(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def main():
    print("Extracting LAB+HSV features...")
    X, y = extract_color_features(CSV_FILE, IMAGE_FOLDER)

    print(f"Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Standardize for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "model/color_scaler.pkl")

    print("Building color MLP...")
    model = build_color_mlp(input_dim=5, num_classes=len(LABEL_MAP))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=8,
        factor=0.5,
        min_lr=1e-5
    )

    print("Training...")
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=300,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )

    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_classes)
    print(f"\n🎯 FINAL COLOR MODEL ACCURACY: {acc * 100:.2f}%\n")

    model.save("model/color_mlp_full.h5")
    print("Saved: model/color_mlp_full.h5")


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    main()

'''

'''
//knn+lab - 82%

import numpy as np
import pandas as pd
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

COLOR_LABEL = "sea-800"   # <--- The label that is purely based on color

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

def extract_color_points(csv_file, image_folder):
    df = pd.read_csv(csv_file)

    X = []
    y = []

    for _, row in df.iterrows():
        label = row['label'].strip().lower()
        if label not in LABEL_MAP:
            continue

        image_path = f"{image_folder}/{row['filename']}"
        img = cv2.imread(image_path)
        if img is None:
            continue

        # Convert to LAB for better color separation
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        pts = row['points'].split(";")
        for p in pts:
            x, ycoord = map(float, p.split(","))
            x, ycoord = int(x), int(ycoord)

            if x < 0 or ycoord < 0 or x >= img.shape[1] or ycoord >= img.shape[0]:
                continue

            pixel = lab[ycoord, x] / 255.0  # L,A,B normalized
            X.append(pixel)
            y.append(LABEL_MAP[label])

    return np.array(X), np.array(y)


def main():
    X, y = extract_color_points("all_annotations.csv", "images")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # KNN works extremely well for color classification
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Color-only model accuracy: {acc * 100:.2f}%")

    # Save model
    import joblib
    joblib.dump(model, "model/color_knn_classifier.pkl")

    print("Saved color_knn_classifier.pkl")

if __name__ == "__main__":
    main()

'''

'''
//88% - cnn but takes texture into account as well
import os
import cv2
import math
import random
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

# -----------------------
# Config
# -----------------------
LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

PATCH_SIZE = 5
HALF = PATCH_SIZE // 2
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 500
MODEL_DIR = "model"
CSV_FILE = "all_annotations.csv"
IMAGE_FOLDER = "images"
AUGMENT_PROB = 0.7  # probability to apply augmentation to a patch

os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -----------------------
# Preprocessing helpers
# -----------------------
def preprocess_image(img):
    """
    Image-level preprocessing:
    - Convert BGR -> HSV
    - Gaussian blur (small)
    - Equalize V channel for brightness normalization
    Returns HSV image (uint8)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    # Equalize V channel
    v = hsv[:, :, 2].astype(np.uint8)
    v_eq = cv2.equalizeHist(v)
    hsv[:, :, 2] = v_eq
    return hsv

def extract_patches_from_row(row, image_folder):
    """
    Given a CSV row, returns list of (patch, label) pairs for valid points.
    """
    filename = row['filename']
    label_name = str(row['label']).strip().lower()
    if label_name not in LABEL_MAP:
        return []

    image_path = os.path.join(image_folder, filename)
    img = cv2.imread(image_path)
    if img is None:
        # missing image
        return []

    hsv_img = preprocess_image(img)  # uint8 HSV

    patches = []
    points = str(row['points']).split(';')
    for pstr in points:
        if not pstr.strip():
            continue
        try:
            x_f, y_f = map(float, pstr.strip().split(','))
        except Exception:
            continue
        x, y = int(x_f), int(y_f)

        # boundary check for full patch
        if x < HALF or y < HALF or x >= hsv_img.shape[1] - HALF or y >= hsv_img.shape[0] - HALF:
            continue

        patch = hsv_img[y-HALF:y+HALF+1, x-HALF:x+HALF+1].astype(np.float32)  # shape (PATCH_SIZE, PATCH_SIZE, 3)
        # normalize to [0,1]
        patch = patch / 255.0
        patches.append((patch, LABEL_MAP[label_name]))

    return patches

def load_all_patches(csv_file, image_folder):
    """
    Loads all patches and labels into memory arrays.
    (This is fine if total patches are reasonably sized. If huge, prefer streaming.)
    """
    df = pd.read_csv(csv_file)
    features = []
    labels = []
    total_rows = len(df)
    print(f"CSV rows: {total_rows}")
    for i, row in df.iterrows():
        patches = extract_patches_from_row(row, image_folder)
        for p, lbl in patches:
            features.append(p)
            labels.append(lbl)
    if len(features) == 0:
        raise RuntimeError("No valid patches found. Check CSV and images.")
    features = np.stack(features, axis=0)
    labels = np.array(labels, dtype=np.int32)
    return features, labels

# -----------------------
# Augmentation utilities
# -----------------------
def random_brightness(patch, max_delta=0.2):
    # patch in [0,1]
    delta = random.uniform(-max_delta, max_delta)
    patch = patch.copy()
    patch[:, :, 2] = np.clip(patch[:, :, 2] + delta, 0.0, 1.0)
    return patch

def random_flip(patch):
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    return patch

def random_rotate_small(patch, max_angle=10):
    # rotate the small patch by a small angle (in degrees)
    angle = random.uniform(-max_angle, max_angle)
    h, w = patch.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    patch_rot = cv2.warpAffine((patch*255).astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    patch_rot = patch_rot.astype(np.float32) / 255.0
    return patch_rot

def augment_patch(patch):
    # apply a random sequence of augmentations
    p = patch
    if random.random() < 0.5:
        p = random_flip(p)
    if random.random() < 0.5:
        p = random_brightness(p, max_delta=0.15)
    if random.random() < 0.3:
        p = random_rotate_small(p, max_angle=8)
    return p

# -----------------------
# Keras Sequence for batches (with augmentation)
# -----------------------
class PatchSequence(Sequence):
    def __init__(self, x, y, batch_size=32, augment=False, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[batch_idx].copy()
        batch_y = self.y[batch_idx].copy()
        if self.augment:
            for i in range(batch_x.shape[0]):
                if random.random() < AUGMENT_PROB:
                    batch_x[i] = augment_patch(batch_x[i])
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# -----------------------
# Model architecture (small CNN for 5x5 patches)
# -----------------------
def build_cnn_patch_model(input_shape, num_classes):
    model = models.Sequential()
    # Input shape e.g. (5,5,3)
    model.add(layers.Input(shape=input_shape))

    # small conv stack
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))  # will reduce 5->2 (floor), ok for tiny patch
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -----------------------
# Main training pipeline
# -----------------------
def main():
    print("Loading patches from CSV...")
    X, y = load_all_patches(CSV_FILE, IMAGE_FOLDER)
    print(f"Loaded patches shape: {X.shape}, labels shape: {y.shape}")

    # Shuffle dataset
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # Standardize per-channel across dataset (flatten over spatial dims)
    # Compute mean/std per feature (i.e., per flattened channel-position)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X = (X - mean) / std

    # Save scaler (mean/std) for later inference
    np.savez(os.path.join(MODEL_DIR, "patch_scaler.npz"), mean=mean, std=std)

    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # compute class weights
    classes = np.unique(y)
    cw_vals = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = {int(classes[i]): float(cw_vals[i]) for i in range(len(classes))}
    print("Class weights:", class_weights)

    # build model
    model = build_cnn_patch_model(input_shape=(PATCH_SIZE, PATCH_SIZE, 3), num_classes=len(LABEL_MAP))
    model.summary()

    # callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # generators
    train_seq = PatchSequence(x_train, y_train, batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_seq = PatchSequence(x_test, y_test, batch_size=BATCH_SIZE, augment=False, shuffle=False)

    # fit
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=2
    )

    # Save model and training history
    model.save(os.path.join(MODEL_DIR, "cnn_patch_classifier.h5"))
    np.savez(os.path.join(MODEL_DIR, "training_history.npz"), 
             history_epoch=np.array(history.epoch),
             history_history=history.history)
    print("Model and artifacts saved to", MODEL_DIR)

if __name__ == "__main__":
    main()

'''


'''
//85%
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models

# Labels used for training
LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

PATCH_SIZE = 3   # Neighborhood patch size
HALF = PATCH_SIZE // 2


# --------------------------------------------------------
# 🔥 FUNCTION: FULL PREPROCESSING PER IMAGE + PIXEL POINTS
# --------------------------------------------------------
def preprocess_image(img):
    """Apply image-level preprocessing."""
    
    # Convert BGR → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Apply Gaussian Blur to reduce noise
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

    # Normalize brightness (Value channel)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2].astype(np.uint8))

    return hsv


def extract_point_features(csv_file, image_folder):
    df = pd.read_csv(csv_file)
    features = []
    labels = []

    for _, row in df.iterrows():
        filename = row['filename']
        label_name = row['label'].strip().lower()

        if label_name not in LABEL_MAP:
            continue

        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARNING] Could not read image: {image_path}")
            continue

        # 🔥 Preprocess image
        hsv_img = preprocess_image(img)

        # Process pixel points
        point_strs = row['points'].split(';')
        for point_str in point_strs:
            x, y = map(float, point_str.strip().split(','))
            x, y = int(x), int(y)

            # Skip invalid points near borders
            if x < HALF or y < HALF or x >= img.shape[1] - HALF or y >= img.shape[0] - HALF:
                continue

            # Extract 3×3 HSV patch
            patch = hsv_img[y-HALF:y+HALF+1, x-HALF:x+HALF+1]
            patch = patch / 255.0
            patch_flat = patch.flatten()

            features.append(patch_flat)
            labels.append(LABEL_MAP[label_name])

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


# --------------------------------------------------------
# 🔥 BUILD ADVANCED MODEL WITH REGULARIZATION
# --------------------------------------------------------
def build_advanced_classifier(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# --------------------------------------------------------
# 🔥 MAIN TRAINING PIPELINE
# --------------------------------------------------------
def main():
    print("📌 Loading dataset...")
    features, labels = extract_point_features('all_annotations.csv', 'images')
    print(f"✔ Total usable training samples: {len(features)}")

    # Dataset-level preprocessing
    print("📌 Standardizing features...")
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features = (features - mean) / std

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Compute class weights
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = {i: cw[i] for i in range(len(cw))}

    # Build model
    model = build_advanced_classifier(
        input_shape=(PATCH_SIZE * PATCH_SIZE * 3,),
        num_classes=len(LABEL_MAP)
    )

    # Early stopping + LR reduction
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )

    print("🚀 Training model with preprocessing + regularization...")
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=500,
        batch_size=32,
        callbacks=[early_stop, lr_schedule],
        class_weight=class_weights
    )

    # Save model
    os.makedirs('model', exist_ok=True)
    model.save('model/advanced_preprocessed_classifier.h5')

    print("🎉 Model saved to model/advanced_preprocessed_classifier.h5")


if __name__ == '__main__':
    main()
'''

'''
//initial code
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

TARGET_SIZE = (256, 256)

LABEL_MAP = {
    "green-800": 0,
    "non-green-800": 1,
    "sea-800": 2,
    "coconut-800": 3
}

def extract_point_features(csv_file, image_folder):
    df = pd.read_csv(csv_file)
    features = []
    labels = []

    for _, row in df.iterrows():
        image_path = os.path.join(image_folder, row['filename'])
        img = cv2.imread(image_path)
        if img is None:
            continue

        point_strs = row['points'].split(';')
        for point_str in point_strs:
            x, y = map(float, point_str.strip().split(','))
            x, y = int(x), int(y)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                rgb = img[y, x] / 255.0
                features.append(rgb)
                label = row['label'].strip().lower()
                labels.append(LABEL_MAP.get(row['label'], -1))

    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def build_simple_classifier(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    features, labels = extract_point_features('all_annotations.csv', 'images')
    mask = labels >= 0
    features, labels = features[mask], labels[mask]

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = build_simple_classifier(input_shape=(3,), num_classes=len(LABEL_MAP))
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=300, batch_size=32)

    os.makedirs('model', exist_ok=True)
    model.save('model/point_classifier.h5')
    print("Model saved to model/point_classifier.h5")

if __name__ == '__main__':
    main()
'''
