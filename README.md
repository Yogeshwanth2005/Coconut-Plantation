# Coconut Plantation — Tree Counting from Aerial Imagery

Counts coconut palms in aerial/satellite imagery. Give it a photo, get a
number and a map of what it found.

```bash
python segmentation/count_trees.py --image plantation.png --save-overlay counted.png
```

```
Image: plantation.png  (536x442)
Stage 1: masking non-vegetation...
  kept 96.8% of pixels as vegetation
Scale: canopy pitch 35px -> resizing 1.61x (trained on ~56px)
Stage 2: segmenting on cpu...

========================================
TREES COUNTED: 157
========================================
Overlay written to counted.png
```

**Always open the overlay.** It draws a ring on every detected tree, and it
is the only reliable way to tell a good count from a confident wrong one.
If the rings sit on palm crowns, trust the number.

---

## Accuracy

Measured as tree count against hand-annotated ground truth, scored only
inside annotated regions.

| | |
|---|---|
| **Detection F1** | **0.86 – 0.92** |
| Precision | 0.82 – 0.90 |
| Recall | 0.90 – 0.93 |
| Count error | +3% to +10% |

These are measured on imagery **held out of training entirely** — the model
had never seen those locations. That is the honest test, and the reason to
trust the figures: it measures generalization to new imagery rather than
memorization. Accuracy on imagery the model *was* trained on falls in the
same range, which is itself a good sign — a model that had memorized its
training set would score far higher there and collapse on everything else.

Errors skew toward slight over-counting rather than missing trees.

### Where it works, and where it doesn't

Built for one job: **top-down aerial or satellite views of coconut
plantations.** Within that, zoom does not matter — scale is detected and
corrected automatically.

It will return a confident, wrong number on:

- ground-level or angled photos (it only ever saw top-down crowns)
- forests, gardens, or other tree species — anything crown-shaped is counted
- buildings with green-ish roofs, which survive vegetation masking
- images so low-resolution the crowns are gone; rescaling corrects zoom, it
  cannot restore detail that was never captured
- scattered trees with no planting grid, where scale detection has no
  periodicity to measure

---

## How it works

```
image → Stage 1 → rescale → tile → Stage 2 → stitch → count
```

1. **Scale detection.** The model learned a canopy whose planting pitch is
   ~56px. The pitch of *your* image is measured directly from the pixels by
   2D autocorrelation — plantations sit on a grid, so that period is
   measurable without any model — and the image is resized to match. This is
   what lets an arbitrary upload work.
2. **Stage 1 — vegetation masking** (`segmentation/stage1_mask.py`). Paints
   over soil, roads and water using Excess Green (`2G-R-B`) with a per-image
   Otsu threshold. Per-image, because exposure varies site to site.
3. **Tiling.** Cut into 256×256 tiles, the model's input size.
4. **Stage 2 — segmentation** (MK-UNet, 6.7M parameters). Marks every pixel
   as tree or background. This is the only learned component.
5. **Stitch, then count.** Tiles are reassembled into one full-size mask
   *before* counting. This matters: a tree on a tile boundary would
   otherwise be counted twice, and ~10% of the area sits near a boundary.

Only step 4 is a neural network. The rest is arithmetic that gets the image
into the shape the model expects and turns pixels into a number.

### A note on the colour algorithm

`Color Algorithm/` holds the project's original Stage 1: a small neural
network that classifies a pixel into **green / non-green / sea / coconut**
from 14 hand-built colour features (Lab, HSV and YCrCb statistics over a 5×5
patch). It is a good classifier and the model files still work.

**It is not on the counting path today**, for a practical reason rather than
a quality one. That classifier labels *individual annotated points*; the
separate script that applied it across every pixel of an image to produce a
mask is not in this repository and predates it. Rebuilding that step from
the classifier was attempted and abandoned — no single threshold reproduced
the original masks across different imagery, and the closest attempt still
undercounted trees by 47% on one location while scoring well on another.

Stage 1 was therefore rebuilt from scratch around a vegetation index, which
needs no missing model and generalizes across locations. The colour
classifier is kept as reference, and the training data it produced is still
what the segmentation model learned from.

Worth knowing if you revisit this: judge any Stage 1 change by the
**end-to-end tree count on more than one location**, never by how closely
the mask resembles the old ones. An earlier rebuild matched the original
masks on 84% of pixels while undercounting trees by 97% — the 16% it got
wrong was precisely the canopy.

### Seeing each step

```bash
python segmentation/count_trees.py --image photo.png --save-steps steps/
```

Writes one image per stage, numbered in order. **`6_stage2_over_image.png`
is the most informative** — it shows the model's raw output over your photo,
and tells you immediately whether the model is finding crowns or guessing.
When a count looks wrong, read `2_stage1_vegetation_only.png` first: a
vegetation mask with holes punched through the canopy looks nothing like a
model failure but produces one.

---

## Command reference

| Flag | Purpose |
|---|---|
| `--image PATH` | Input image (required) |
| `--save-overlay PATH` | Write the image with a ring per tree |
| `--save-steps DIR` | Write one image per pipeline stage |
| `--save-mask PATH` | Write the raw binary tree mask |
| `--scale N` | Override automatic scale detection |
| `--skip-stage1` | Input is already vegetation-masked |
| `--checkpoint PATH` | Use a different model file |

Run from the project root, and quote paths containing spaces.

---

## Training

The model is trained on Modal (GPU). The data pipeline runs in order:

```bash
python segmentation/generate_blob_masks.py    # annotations → masks
python segmentation/tile_and_split.py         # → 256px tiles, site-aware split
modal run segmentation/modal_train_segmentation.py
modal volume get coconut-segmentation-output mkunet_binary_best.pth . --force
```

**Data:** 6 geographically distinct locations, 18 annotated source tiles at
8192×4283, ~21,700 marked tree points. Two locations are held out entirely
so generalization to unseen imagery can be measured honestly. After
filtering, 1,305 training / 230 validation / 818 test tiles.

The split is **location-aware, never random.** Randomly shuffling tiles
would put crops of the same source image in both training and test, and the
resulting score would measure memorization rather than generalization.

### The ignore mask — the thing that made this work

Masks carry **three** values, not two: `0` background, `255` tree, and
**`128` = never annotated**.

Annotation boxes cover only ~15% of each source image. Before the ignore
band existed, everything outside a box was labeled background — which meant
**50% of all training tiles were dense, unmistakable coconut canopy labeled
"not a tree."** The model was being trained against itself on half its data.

Adding the ignore band, so unlabeled pixels are excluded from both the loss
and the metrics, took held-out accuracy from **F1 0.338 to 0.918** — the
single largest improvement in the project's history.

Anything reading a mask must treat `128` as unlabeled. Never `mask > 0`,
never `count_nonzero` for tree coverage.

### Evaluation

```bash
# count-based metrics on the held-out split
python segmentation/count_predictions.py --checkpoint mkunet_binary_best.pth --split test

# side-by-side visual review panels
python segmentation/inspect_predictions.py --checkpoint mkunet_binary_best.pth --split test --n 12
```

Both accept `--site PREFIX` to score one location at a time.

Judge this pipeline by **count metrics, not pixel IoU.** IoU sits around
0.32 while count F1 is 0.918, and that gap is expected: the training masks
are fixed-radius circles at annotated points, not true canopy outlines, so
exact pixel overlap was never the target.

---

## Layout

| Path | What it is |
|---|---|
| `segmentation/count_trees.py` | **The product.** Image → count |
| `segmentation/stage1_mask.py` | Vegetation masking |
| `segmentation/generate_blob_masks.py` | Annotations → training masks |
| `segmentation/tile_and_split.py` | Tiling + site-aware split |
| `segmentation/modal_train_segmentation.py` | Training (Modal GPU) |
| `segmentation/count_predictions.py` | Count-based evaluation |
| `segmentation/inspect_predictions.py` | Visual prediction review |
| `Applicatno/MK-UNet-main/` | Model architecture and dataloader |
| `Applicatno/annotations/` | Point annotations (JSON) |
| `Coconut/` | Source imagery, by site |
| `HarinieColourAlgo/` | Colour classifier — the original Stage 1; see the note above |
| `docs/superpowers/specs/` | Design spec |
| `.agents/` | Project knowledge base — decisions, gotchas, backlog |

Superseded and kept only for reference: `patch classification/`,
`Applicatno/MK-UNet-main/train_plantation.py`, and `pipeline.py` at the root
(which still runs the old ResNet-18 path). Don't extend these.

## Requirements

Python 3.11+, `torch`, `opencv-python`, `numpy`, `scipy`, `timm`,
`albumentations`. Training additionally needs `modal`. A GPU is optional —
inference runs on CPU, taking seconds for a small photo and ~10 minutes for
a full 8192×4283 tile.

## Status

Counting (design spec Goal 1) is complete. Characterizing **planting
arrangement** (Goal 2) and handling **mixed arrangement styles** (Goal 3)
are not started. Current backlog lives in
`.agents/projects/active-backlog.md`; the reasoning behind past decisions,
including several traps worth not rediscovering, is in
`.agents/decisions/log.md`.
