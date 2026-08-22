"""
Uploads Applicatno/Patched_data_split (all 15 density classes, train+test
pooled) into a Modal Volume so training runs can read it on Modal's GPUs.

Run once (or whenever the local dataset changes):
    modal run "patch classification/modal_upload.py"
"""

import modal
from pathlib import Path

app = modal.App("coconut-density-upload")

volume = modal.Volume.from_name("coconut-patch-data", create_if_missing=True)

LOCAL_DATA_ROOT = Path(__file__).resolve().parent.parent / "Applicatno" / "Patched_data_split"
REMOTE_DATA_ROOT = "/data/raw"

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@app.local_entrypoint()
def main():
    if not LOCAL_DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found at {LOCAL_DATA_ROOT}")

    files = [
        p for split_dir in ("train", "test")
        for p in (LOCAL_DATA_ROOT / split_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    print(f"Found {len(files)} images under {LOCAL_DATA_ROOT}")

    with volume.batch_upload(force=True) as batch:
        for i, f in enumerate(files):
            # Pool train+test; class = parent folder name (e.g. "1".."15")
            cls = f.parent.name
            remote_path = f"{REMOTE_DATA_ROOT}/{cls}/{f.name}"
            batch.put_file(f, remote_path)
            if (i + 1) % 500 == 0:
                print(f"  uploaded {i + 1}/{len(files)}")

    print(f"Upload complete: {len(files)} images -> volume 'coconut-patch-data' at {REMOTE_DATA_ROOT}")
