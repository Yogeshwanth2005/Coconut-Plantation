"""
Evaluates the trained ResNet-18 density classifier on the held-out
validation split (same box-grouped stratified split used in
modal_train.py, same SEED, so this is the exact set the model never
trained on) and reports per-class precision/recall/F1 + a confusion
matrix.

Run:
    modal run "patch classification/modal_eval.py"
"""

import re
from collections import defaultdict

import modal

app = modal.App("coconut-density-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "scikit-learn==1.5.2",
        "numpy<2",
    )
)

data_volume = modal.Volume.from_name("coconut-patch-data", create_if_missing=True)
output_volume = modal.Volume.from_name("coconut-model-output", create_if_missing=True)

VOLUME_MOUNT = "/data"
DATA_ROOT = "/data/data/raw"
OUTPUT_ROOT = "/output"

NUM_CLASSES = 15
IMG_SIZE = 128  # must match modal_train.py's IMG_SIZE for the checkpoint being evaluated
BATCH_SIZE = 32
VAL_FRACTION = 0.2
SEED = 42  # must match modal_train.py so this is the same held-out split


def box_id_from_filename(name: str) -> str:
    return re.sub(r"_\d+_.*$", "", name)


@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_MOUNT: data_volume, OUTPUT_ROOT: output_volume},
    timeout=30 * 60,
)
def evaluate():
    import random
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from PIL import Image
    from sklearn.metrics import classification_report, confusion_matrix

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(DATA_ROOT)
    split_dirs = [p for p in (data_root / "train", data_root / "test") if p.is_dir()]
    class_roots = split_dirs if split_dirs else [data_root]

    class_names = sorted(
        {p.name for root in class_roots for p in root.iterdir() if p.is_dir()},
        key=lambda x: int(x),
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    groups = defaultdict(list)
    for cls in class_names:
        for root in class_roots:
            cls_dir = root / cls
            if not cls_dir.is_dir():
                continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                box_id = box_id_from_filename(f.name)
                groups[(cls, box_id)].append(f)

    # ---- Rebuild the identical val split used in training ----
    val_files, val_labels = [], []
    by_class = defaultdict(list)
    for (cls, box_id), files in groups.items():
        by_class[cls].append((box_id, files))

    for cls, box_groups in by_class.items():
        rng = random.Random(SEED)
        rng.shuffle(box_groups)
        n_val_boxes = max(1, round(len(box_groups) * VAL_FRACTION))
        val_boxes = box_groups[:n_val_boxes]
        for _, files in val_boxes:
            val_files.extend(files)
            val_labels.extend([class_to_idx[cls]] * len(files))

    print(f"Held-out validation images: {len(val_files)}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class PatchDataset(Dataset):
        def __init__(self, files, labels, transform):
            self.files, self.labels, self.transform = files, labels, transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img), self.labels[idx], idx

    val_ds = PatchDataset(val_files, val_labels, transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    ckpt_path = Path(OUTPUT_ROOT) / "resnet18_density_classifier.pth"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds, all_labels, all_indices = [], [], []
    with torch.no_grad():
        for images, labels, indices in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_indices.extend(indices.numpy())

    overall_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    print(f"\nOverall accuracy: {overall_acc:.4f}\n")

    # ---- Dump misclassified files for specific classes of interest ----
    classes_of_interest = {"6", "11", "12", "13"}
    print("Misclassified files (true -> pred) for classes of interest:")
    for pred, true, idx in zip(all_preds, all_labels, all_indices):
        true_name = class_names[true]
        pred_name = class_names[pred]
        if pred != true and (true_name in classes_of_interest or pred_name in classes_of_interest):
            print(f"  {val_files[idx].name}  [true={true_name} pred={pred_name}]")

    report = classification_report(
        all_labels, all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0,
        digits=3,
    )
    print("Per-class report:")
    print(report)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    print("Confusion matrix (rows=true, cols=pred):")
    header = "      " + " ".join(f"{c:>4}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>5} " + " ".join(f"{v:>4d}" for v in row))

    return {"overall_acc": overall_acc, "report": report}


@app.local_entrypoint()
def main():
    evaluate.remote()
