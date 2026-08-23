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
  - Input size raised 64->128: source patches are natively 512x512, and
    the row-spacing detail that separates classes 11/12/13 doesn't
    survive a 64x64 downscale (confirmed by visual inspection of
    misclassified patches).
  - Resolution-degradation augmentation: a fraction of training images
    are blurred/downsampled-then-upsampled each epoch, so the model
    stays robust to lower-quality real-world source imagery, not just
    today's 8192x4283 tiles.

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

VOLUME_MOUNT = "/data"
DATA_ROOT = "/data/data/raw"  # matches `modal volume put coconut-patch-data <local> /data/raw`
OUTPUT_ROOT = "/output"

NUM_CLASSES = 15
IMG_SIZE = 128
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
VAL_FRACTION = 0.2
SEED = 42

# Resolution-degradation augmentation: simulates lower-quality source
# imagery so the model doesn't only work well on today's high-res tiles.
DEGRADE_PROB = 0.35          # fraction of training images degraded per epoch
DEGRADE_MIN_SCALE = 0.25     # degraded down to as little as 25% of IMG_SIZE...
DEGRADE_MAX_SCALE = 0.75     # ...up to 75%, then upsampled back to IMG_SIZE


def box_id_from_filename(name: str) -> str:
    # "AM-1-E-1_0_original.png" -> "AM-1-E-1"
    return re.sub(r"_\d+_.*$", "", name)


class RandomResolutionDegrade:
    """Simulate a lower-quality source photo: shrink then blow back up.

    A picklable class (not a closure) so it works safely with DataLoader
    worker processes under either 'fork' or 'spawn'.
    """

    def __init__(self, img_size, prob, min_scale, max_scale):
        self.img_size = img_size
        self.prob = prob
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        import random as _random
        from PIL import Image as _Image

        if _random.random() >= self.prob:
            return img
        scale = _random.uniform(self.min_scale, self.max_scale)
        small_size = max(8, int(self.img_size * scale))
        img = img.resize((small_size, small_size), _Image.BILINEAR)
        return img.resize((self.img_size, self.img_size), _Image.BILINEAR)


class FixedResolutionDegrade:
    """Always shrink to a fixed scale then blow back up (for eval)."""

    def __init__(self, img_size, scale):
        self.img_size = img_size
        self.scale = scale

    def __call__(self, img):
        from PIL import Image as _Image

        small_size = max(8, int(self.img_size * self.scale))
        img = img.resize((small_size, small_size), _Image.BILINEAR)
        return img.resize((self.img_size, self.img_size), _Image.BILINEAR)


@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_MOUNT: data_volume, OUTPUT_ROOT: output_volume},
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

    # Data may be laid out flat (data_root/<class>/) or pooled under
    # train/test subfolders (data_root/{train,test}/<class>/) -- support both.
    split_dirs = [p for p in (data_root / "train", data_root / "test") if p.is_dir()]
    class_roots = split_dirs if split_dirs else [data_root]

    class_names = sorted(
        {p.name for root in class_roots for p in root.iterdir() if p.is_dir()},
        key=lambda x: int(x),
    )
    assert len(class_names) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes, found {len(class_names)}: {class_names}"
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # ---- Group files by (class, box_id) so augmented siblings move together ----
    groups = defaultdict(list)  # (class, box_id) -> [file paths]
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        RandomResolutionDegrade(IMG_SIZE, DEGRADE_PROB, DEGRADE_MIN_SCALE, DEGRADE_MAX_SCALE),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    # Degraded-val: same held-out images, but always shrunk to the low end
    # of the degradation range -- measures robustness, not just clean accuracy.
    val_degraded_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        FixedResolutionDegrade(IMG_SIZE, DEGRADE_MIN_SCALE),
        transforms.ToTensor(),
        normalize,
    ])

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
    val_degraded_ds = PatchDataset(val_files, val_labels, val_degraded_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    val_degraded_loader = DataLoader(val_degraded_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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

    # ---- Evaluate the best checkpoint on degraded (low-res-simulated) val images ----
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_degraded_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    degraded_val_acc = correct / total
    print(f"Degraded-val accuracy (robustness check): {degraded_val_acc:.4f}")

    out_path = Path(OUTPUT_ROOT) / "resnet18_density_classifier.pth"
    torch.save(best_state, out_path)
    output_volume.commit()
    print(f"Saved best checkpoint to {out_path}")

    return {
        "best_val_acc": best_val_acc,
        "degraded_val_acc": degraded_val_acc,
        "class_names": class_names,
    }


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"\nTraining finished.")
    print(f"  Clean val accuracy:    {result['best_val_acc']:.4f}")
    print(f"  Degraded val accuracy: {result['degraded_val_acc']:.4f}")
    print("Download with:")
    print("  modal volume get coconut-model-output resnet18_density_classifier.pth . --force")
