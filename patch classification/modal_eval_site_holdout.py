"""
Diagnostic: quantifies whether the density classifier is relying on
site-specific color/lighting as a shortcut instead of genuine
density/pattern signal.

Most classes in Patched_data_split map almost entirely to a single
plantation site (e.g. class 1=Amrita only, class 12=Wat Phleng only),
so a random train/val split can't tell color-shortcut apart from real
learning for those classes. Only 5 classes span two sites (5, 7, 11,
13, 14 -- all Kradangnga(KR) + Wat Phleng(WA)), so this script:

  1. Trains a fresh model where, for those 5 classes ONLY, one site's
     boxes are held out entirely from training and used as a same-site
     val set... no wait -- held out and used as a CROSS-site test set.
  2. Reports accuracy on that cross-site test set for those 5 classes
     specifically, compared to a normal same-distribution val accuracy.
  3. A large gap between "same-site" and "cross-site" accuracy on
     these classes is direct evidence of the color/site shortcut.

All other classes train normally (regular box-grouped stratified
split) since there's no second site to test them against -- this is
a diagnostic, not a replacement for modal_train.py.

Run:
    modal run "patch classification/modal_eval_site_holdout.py"
"""

import re
from collections import defaultdict

import modal

app = modal.App("coconut-density-site-holdout")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy<2",
    )
)

data_volume = modal.Volume.from_name("coconut-patch-data", create_if_missing=True)

VOLUME_MOUNT = "/data"
DATA_ROOT = "/data/data/raw"

NUM_CLASSES = 15
IMG_SIZE = 128
BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 12
LEARNING_RATE = 1e-4
VAL_FRACTION = 0.2  # for the normal (non-cross-site) classes
SEED = 42

# Classes with boxes from more than one site -- the only ones we can
# fairly test for cross-site generalization.
CROSS_SITE_CLASSES = {"5", "7", "11", "13", "14"}
# Held out entirely from training for those classes; used as the
# cross-site test set. The other site's boxes for these classes stay
# in training (split the normal way).
HOLDOUT_SITE = "WA"


def box_id_from_filename(name: str) -> str:
    return re.sub(r"_\d+_.*$", "", name)


def site_from_box_id(box_id: str) -> str:
    return box_id.split("-")[0]


@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_MOUNT: data_volume},
    timeout=60 * 60,
)
def run():
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
    split_dirs = [p for p in (data_root / "train", data_root / "test") if p.is_dir()]
    class_roots = split_dirs if split_dirs else [data_root]

    class_names = sorted(
        {p.name for root in class_roots for p in root.iterdir() if p.is_dir()},
        key=lambda x: int(x),
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

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

    by_class = defaultdict(list)  # cls -> [(box_id, files)]
    for (cls, box_id), files in groups.items():
        by_class[cls].append((box_id, files))

    train_files, train_labels = [], []
    val_files, val_labels = [], []          # normal same-distribution val
    cross_site_files, cross_site_labels = [], []  # held-out-site test

    for cls, box_groups in by_class.items():
        rng = random.Random(SEED)
        rng.shuffle(box_groups)

        if cls in CROSS_SITE_CLASSES:
            # Held-out site's boxes -> cross-site test only, never trained on.
            # Other site's boxes -> normal stratified train/val split.
            other_site_groups = [
                (bid, files) for bid, files in box_groups
                if site_from_box_id(bid) != HOLDOUT_SITE
            ]
            holdout_groups = [
                (bid, files) for bid, files in box_groups
                if site_from_box_id(bid) == HOLDOUT_SITE
            ]
            for _, files in holdout_groups:
                cross_site_files.extend(files)
                cross_site_labels.extend([class_to_idx[cls]] * len(files))

            n_val = max(1, round(len(other_site_groups) * VAL_FRACTION)) if other_site_groups else 0
            val_boxes = other_site_groups[:n_val]
            train_boxes = other_site_groups[n_val:]
            for _, files in train_boxes:
                train_files.extend(files)
                train_labels.extend([class_to_idx[cls]] * len(files))
            for _, files in val_boxes:
                val_files.extend(files)
                val_labels.extend([class_to_idx[cls]] * len(files))
        else:
            n_val = max(1, round(len(box_groups) * VAL_FRACTION))
            val_boxes = box_groups[:n_val]
            train_boxes = box_groups[n_val:]
            for _, files in train_boxes:
                train_files.extend(files)
                train_labels.extend([class_to_idx[cls]] * len(files))
            for _, files in val_boxes:
                val_files.extend(files)
                val_labels.extend([class_to_idx[cls]] * len(files))

    print(f"Train images: {len(train_files)}")
    print(f"Same-distribution val images: {len(val_files)}")
    print(f"Cross-site ({HOLDOUT_SITE}-held-out) test images: {len(cross_site_files)}")
    print(f"Cross-site test covers classes: {sorted(CROSS_SITE_CLASSES, key=int)}")
    if not cross_site_files:
        print("WARNING: no cross-site test images found -- check HOLDOUT_SITE / data.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    class PatchDataset(Dataset):
        def __init__(self, files, labels, transform):
            self.files, self.labels, self.transform = files, labels, transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]

    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = class_counts.sum() / (NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        PatchDataset(train_files, train_labels, transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        PatchDataset(val_files, val_labels, transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )
    cross_site_loader = None
    if cross_site_files:
        cross_site_loader = DataLoader(
            PatchDataset(cross_site_files, cross_site_labels, transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        )

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
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Epoch {epoch + 1:3d}/{MAX_EPOCHS}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    print(f"\nBest same-distribution val accuracy: {best_val_acc:.4f}")

    if cross_site_loader is not None:
        # Overall cross-site accuracy
        correct, total = 0, 0
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        with torch.no_grad():
            for images, labels in cross_site_loader:
                images_dev = images.to(device)
                preds = model(images_dev).argmax(1).cpu()
                for p, l in zip(preds, labels):
                    correct += int(p == l)
                    total += 1
                    cls_name = class_names[l.item()]
                    per_class_total[cls_name] += 1
                    per_class_correct[cls_name] += int(p == l)

        cross_site_acc = correct / total
        print(f"\nCross-site ({HOLDOUT_SITE}-held-out) test accuracy: {cross_site_acc:.4f}")
        print("Per-class cross-site accuracy (classes tested here have 2+ sites):")
        for cls in sorted(CROSS_SITE_CLASSES, key=int):
            t = per_class_total.get(cls, 0)
            c = per_class_correct.get(cls, 0)
            if t:
                print(f"  class {cls:>3}: {c}/{t} = {c / t:.3f}")
            else:
                print(f"  class {cls:>3}: no test examples")

        # Same-distribution val accuracy restricted to just these classes,
        # for a fair apples-to-apples comparison.
        same_dist_correct, same_dist_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                mask = torch.tensor([class_names[l.item()] in CROSS_SITE_CLASSES for l in labels])
                if not mask.any():
                    continue
                images_dev = images.to(device)
                preds = model(images_dev).argmax(1).cpu()
                same_dist_correct += (preds[mask] == labels[mask]).sum().item()
                same_dist_total += mask.sum().item()
        if same_dist_total:
            print(
                f"\nSame-distribution val accuracy, RESTRICTED to classes "
                f"{sorted(CROSS_SITE_CLASSES, key=int)}: "
                f"{same_dist_correct}/{same_dist_total} = {same_dist_correct/same_dist_total:.4f}"
            )
            print(
                "Compare this to the cross-site accuracy above -- a large drop "
                "means the model is relying on site-specific cues (color/lighting) "
                "rather than genuine density/pattern signal."
            )


@app.local_entrypoint()
def main():
    run.remote()
