"""
Trains the ResNet-18 patch-density classifier on Modal, reading data
from the 'coconut-patch-data' Volume (see modal_upload.py) and writing
the best checkpoint back to the 'coconut-model-output' Volume.

Fixes vs. the original Colab notebook (patch classification/resnet-18.ipynb):
  - Train/val split is grouped by source box ID so augmented siblings
    (e.g. AM-1-E-1_original / AM-1-E-1_lr_flip / ...) never appear on
    both sides -- the previous split leaked across all 15 classes.
  - Split is stratified by class, not a single random split.
  - Loss is class-weighted to counter the class imbalance (e.g. class 3
    has ~480 train images, class 15 has ~32).
  - Best-val-accuracy checkpoint is saved, not just whatever the final
    epoch happens to produce.
  - Early stopping, since train accuracy hits ~100% by epoch 6-8.

Run:
    modal run "patch classification/modal_train.py"

Then download the result:
    modal volume get coconut-model-output resnet18_density_classifier.pth .
"""

import re
from collections import defaultdict

import modal

app = modal.App("coconut-density-train")

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

DATA_ROOT = "/data/raw"
OUTPUT_ROOT = "/output"

NUM_CLASSES = 15
IMG_SIZE = 64
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
VAL_FRACTION = 0.2
SEED = 42


def box_id_from_filename(name: str) -> str:
    # "AM-1-E-1_0_original.png" -> "AM-1-E-1"
    return re.sub(r"_\d+_.*$", "", name)


@app.function(
    image=image,
    gpu="T4",
    volumes={DATA_ROOT: data_volume, OUTPUT_ROOT: output_volume},
    timeout=60 * 60 * 2,
)
def train():
    import random
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from PIL import Image

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(DATA_ROOT)
    class_names = sorted(
        (p.name for p in data_root.iterdir() if p.is_dir()),
        key=lambda x: int(x),
    )
    assert len(class_names) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes, found {len(class_names)}: {class_names}"
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # ---- Group files by (class, box_id) so augmented siblings move together ----
    groups = defaultdict(list)  # (class, box_id) -> [file paths]
    for cls in class_names:
        for f in (data_root / cls).iterdir():
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            box_id = box_id_from_filename(f.name)
            groups[(cls, box_id)].append(f)

    # ---- Stratified split at the box-group level ----
    train_files, val_files = [], []
    train_labels, val_labels = [], []

    by_class = defaultdict(list)
    for (cls, box_id), files in groups.items():
        by_class[cls].append((box_id, files))

    for cls, box_groups in by_class.items():
        rng = random.Random(SEED)
        rng.shuffle(box_groups)
        n_val_boxes = max(1, round(len(box_groups) * VAL_FRACTION))
        val_boxes = box_groups[:n_val_boxes]
        train_boxes = box_groups[n_val_boxes:]

        for _, files in train_boxes:
            train_files.extend(files)
            train_labels.extend([class_to_idx[cls]] * len(files))
        for _, files in val_boxes:
            val_files.extend(files)
            val_labels.extend([class_to_idx[cls]] * len(files))

    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_files)}  |  Val images: {len(val_files)}")
    for cls in class_names:
        n_train = sum(1 for l in train_labels if l == class_to_idx[cls])
        n_val = sum(1 for l in val_labels if l == class_to_idx[cls])
        print(f"  class {cls:>3}: train={n_train:4d}  val={n_val:4d}")

    # ---- Class weights (inverse frequency, from train split only) ----
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = class_counts.sum() / (NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(2)}")

    # ---- Dataset ----
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = train_transform  # augmentation already baked into the dataset

    class PatchDataset(Dataset):
        def __init__(self, files, labels, transform):
            self.files = files
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]

    train_ds = PatchDataset(train_files, train_labels, train_transform)
    val_ds = PatchDataset(val_files, val_labels, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ---- Model: same modified ResNet-18 pipeline.py expects ----
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss, train_acc = running_loss / total, correct / total

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_loss, val_acc = running_loss / total, correct / total
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1:3d}/{MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    out_path = Path(OUTPUT_ROOT) / "resnet18_density_classifier.pth"
    torch.save(best_state, out_path)
    output_volume.commit()
    print(f"Saved best checkpoint to {out_path}")

    return {"best_val_acc": best_val_acc, "class_names": class_names}


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"\nTraining finished. Best val accuracy: {result['best_val_acc']:.4f}")
    print("Download with:")
    print("  modal volume get coconut-model-output resnet18_density_classifier.pth .")
