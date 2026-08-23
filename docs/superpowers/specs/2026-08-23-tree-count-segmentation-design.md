# Tree counting via binary segmentation — design spec

**Date:** 2026-08-23
**Status:** Approved, not yet implemented

## Background

The project counts coconut trees in large aerial tiles (8192×4283 px, 6
plantation sites, 18 tiles, ~26,000 hand-annotated tree points across 457
labeled boxes). The pipeline has three intended stages: (1) separate
coconut-canopy-green from everything else, (2) determine density/pattern
within canopy regions, (3) turn that into a tree-count estimate.

Stage 1 (`HarinieColourAlgo`, a color-feature MLP) works and is not
changing.

Stage 2 has been attempted twice, and this spec replaces both attempts:

- **ResNet-18 patch classifier** (`patch classification/`, `pipeline.py`):
  classifies a 128×128 patch into 1 of 15 hand-assigned density/arrangement
  classes. A rigorous cross-site holdout test (train on Kradangnga+Wat
  Phleng examples of the 5 classes that span both sites, test on the
  held-out site) showed accuracy collapsing from 92.5% (same-distribution
  validation) to 7.2% (unseen site) — the model was substantially learning
  per-site color/lighting fingerprints, not density patterns, because most
  of the 15 classes are backed by a single site's data.
- **MK-UNet segmentation** (`Applicatno/MK-UNet-main/`): reuses the same
  15-class scheme (same site-skew problem), trains on masks that are solid
  filled rectangles (the annotation bounding box), not real tree/canopy
  shapes, and its reported ~99% Dice was recomputed by hand to be ~23-27%
  once cells where a class is trivially absent from a tile are excluded.
  Its train/test split is a random shuffle of 256×256 tiles pooled across
  all sites with no site-awareness at all — worse leakage than the
  ResNet-18 model's original (pre-fix) split.

Both failures share a root cause: **the 15-class label conflates two
different signals — tree count and planting arrangement pattern — and
because classes are mostly single-site, the model can't tell "site" apart
from "class."** Additionally, neither approach can handle a single patch
containing multiple different arrangement styles (e.g. half ordered rows,
half scattered), since both produce one label for the whole patch/tile
region.

## Goals

1. Produce an actual tree **count** per patch/region — a number, not a
   classification bucket.
2. Separately characterize planting **arrangement pattern** (e.g. ordered
   rows vs. scattered) without conflating it with count.
3. Handle patches containing mixed arrangement styles.
4. Generalize across plantation sites — validated with a genuine
   site-holdout test, not a random split.

## Non-goals

- Re-attempting the 15-class scheme in any form.
- New manual annotation work (outline tracing, re-labeling). The design
  works entirely from data that already exists (box + point annotations).
- Real-time/low-latency inference. This runs as an offline batch pipeline.

## Assumption: annotation quality is the ceiling

Mask generation (stage 1 below) draws a blob at every point in the
existing annotations and trusts it completely — it has no independent way
to verify that a given point was placed on an actual coconut tree rather
than a different tree species that happened to look similar in the photo.
This is safe only because the original point-marking process was a
deliberate coconut-vs-other-tree visual judgment, not a looser "this looks
green" click — confirmed for the existing 26,000+ points used to build
this pipeline.

This means the ceiling on the whole pipeline's real-world accuracy is set
by the quality of point annotation, both for the data used to train the
segmentation model and for any future data (new sites, new imagery) used
to extend or validate it. If new training data is ever added, it must be
marked with the same care — the pipeline provides no mechanism to catch a
mismarked point on its own.

**Known gap — annotation completeness, not just correctness.** Visual
review of generated masks (`segmentation/masks/`, checked against source
tiles in `Coconut/<Site>/`) confirmed that some real trees inside
annotated boxes were not marked with a point at all, in certain sites/
regions — the boxes themselves are correct, but not exhaustive; not every
tree within a marked box necessarily has a corresponding point. This is a
false-negative gap in the ground truth (a real tree, no point, so no
blob), not a false-positive one (no evidence yet of a point placed where
no tree exists).

There is currently no independent way to quantify how many trees are
missed this way — doing so algorithmically would require the same kind of
tree-detection capability this project is trying to build, a circular
dependency. Decision: proceed with the existing annotations as-is. Expect
this to cause the segmentation model to systematically undercount by some
unknown but likely small margin, inherited directly from the training
data. Revisit by re-checking/re-marking specific boxes only if this
undercount turns out to matter for the accuracy the final pipeline needs
to hit — not a blocker for starting model training.

## Architecture

Three stages, replacing the ResNet-18 classifier and the current MK-UNet
training entirely. Stage 1 (color classification) is unchanged and sits
upstream of this.

### 1. Mask generation (new, no training)

For every annotated tree point, draw a small filled circle onto an
otherwise-blank mask. Everything else is background. This replaces the
existing box-fill mask generation (`json_to_masks.py` /
`remap_categories.py`, which fill the entire annotation bounding box solid
— discarding the individual point locations and producing masks that are
just redrawn rectangles, not tree shapes).

- **Blob radius**: fixed, uniform for every tree (not derived per-tree from
  local spacing — an earlier variable-radius idea was rejected because
  local spacing noise in high-count boxes would distort blob size and
  corrupt exactly the classes most reliant on a clean signal).
- **Radius value**: derived from the actual nearest-neighbor spacing
  distribution across all 26,494 annotated points (computed at full
  8192×4283 resolution): p5 = 36.1px, p50 = 49.7px, mean = 53.0px. A radius
  of **~12–15px** keeps blobs at roughly a third of the tightest observed
  real-world spacing (p5), so even densely-packed trees get distinct,
  non-overlapping blobs.
- **Output**: binary mask per source tile (tree = 1, background = 0),
  same resolution as the source tile, tiled the same way the existing
  `tile_dataset.py` tiles images (256×256, adjust if needed) for training.

This step is pure image processing on data that already exists
(`Applicatno/annotations/*.json`, `Points` arrays) — no model training
required, can be implemented and run immediately.

### 2. Segmentation model (binary, rebuilt MK-UNet)

Reuses the existing MK-UNet network architecture
(`Applicatno/MK-UNet-main/mkunet_network.py`, `MK_UNet_ShallowDec_L`) — the
architecture itself was never the problem; the labels, masks, and split
were. Retrained with:

- **Binary output** (tree vs. background), not 15/16/32 classes. This
  directly removes the site-confound: "is this pixel part of a tree
  canopy" does not depend on which site the photo is from the way "which
  of 15 site-skewed arrangement classes" did.
- **New blob masks** from step 1, not box-fill masks.
- **Site-aware train/test split**: entire sites held out, not a random
  shuffle of tiles. `tile_dataset.py`'s current split
  (`random.shuffle` + ratio slicing across all sites pooled together) must
  be replaced with a split that groups by source site, mirroring the
  box-grouped/site-aware evaluation methodology already built for the
  ResNet-18 model this session (`patch classification/modal_train.py`,
  `modal_eval_site_holdout.py`).
- **Fixed train/eval code bugs** found in the current MK-UNet scripts:
  `train_plantation.py` defaults `--num_classes 16`,
  `test_plantation.py` defaults `--num_classes 32` — must match. The
  eval script's classification-head columns were silently missing from
  output, likely from a `strict=False` state-dict load silently dropping
  a mismatched head — must be fixed or removed if unused.
- **Training infrastructure**: Modal (already set up this session for the
  ResNet-18 model), replacing the original Colab + Google Drive
  checkpoint-sync workflow (`--drive_save_path`). Needs a fresh Modal
  Volume for the new blob-mask dataset, and an updated/pinned dependency
  set (`Applicatno/MK-UNet-main/requirements.txt` currently pins ancient
  torch 1.11/CUDA 11.3 and several unused medical-imaging packages —
  `simpleitk`, `nibabel`, `medpy`, `h5py`, `mmcv` — that can be dropped;
  `mmcv` is never actually imported anywhere in the plantation scripts).

### 3. Downstream analysis (deterministic, not trained)

Both derived from the same binary segmentation output:

- **Count**: connected-component count of blobs in the predicted binary
  mask = tree count for that patch/region. Directly answers "how many
  trees," with no bucket/classification ambiguity, and works on
  mixed-arrangement patches since it's computed per-pixel/per-region
  rather than judged once for the whole patch.
- **Pattern**: geometric analysis of blob centroid positions (e.g.
  regularity of nearest-neighbor spacing and angles, to distinguish
  row-ordered from scattered arrangement). Implemented as a **hand-built
  rule**, not a trained classifier — deliberately, to avoid reintroducing
  the site-overfitting failure mode this spec exists to fix. Works
  directly on the clean blob-position data rather than raw photo
  appearance, which is a smaller and more tractable generalization problem
  than learning "arrangement" from pixel color/texture. Can be upgraded to
  a trained model later if the rules prove insufficient once real output
  is reviewed — not a decision to make speculatively now.

## Evaluation

Two separate honest metrics, replacing the old "accuracy against a 15-way
bucket" evaluation entirely:

- **Count accuracy**: mean absolute error (and/or percentage error)
  between predicted tree count and true annotated point count, per patch.
  A prediction that's off by 2 trees now shows up as "off by 2," not as a
  total miss the way a wrong bucket would.
- **Pattern accuracy**: agreement between the rule's row/scattered call
  and manual visual judgment on a sample of patches (same kind of visual
  review used earlier this session for the disputed ResNet-18 boxes).

Both must be evaluated with a **site-holdout split** (train on some sites,
test on a fully unseen site), not a random split — this is the test that
exposed both prior approaches' failures, and is the only way to honestly
claim the new pipeline generalizes.

## What happens to existing work

- **Retired**: the ResNet-18 15-class density classifier
  (`patch classification/`, its Modal training/eval scripts, the
  `Patched_data_split` dataset as a training target) and the current
  MK-UNet training (box-fill masks, 15/16/32-class scheme,
  `tile_dataset.py`'s random split). Code and trained checkpoints can stay
  in the repo for reference but are no longer part of the active pipeline.
- **Kept**: Stage 1 color classification, all raw annotation data (boxes +
  points — this design derives new masks from the same source data, no
  data is discarded), the Modal training/evaluation infrastructure and
  patterns built this session (box-grouping to prevent augmentation
  leakage, site-aware holdout testing), the MK-UNet network architecture
  code itself.
- **`pipeline.py`** (the root orchestration script) will need Stage 2/3
  rewired to call the new segmentation + count + pattern flow instead of
  the ResNet-18 patch classifier, once the new pipeline exists and is
  validated.

## Open questions / deferred decisions

- Exact blob radius (12–15px range given, final value to be picked when
  implementing based on visual inspection of generated masks).
- Exact geometric rule/thresholds for row vs. scattered classification —
  to be defined during implementation, informed by looking at real blob
  layouts from both known arrangement styles.
- Whether tile size for segmentation training stays at 256×256 (current
  MK-UNet default) or changes.
- Whether/when to revisit a trained pattern classifier if the rule-based
  approach proves insufficient.

## Implementation order

1. Mask generation script (blob masks from existing point annotations) —
   no training required, can be validated visually immediately.
2. Site-aware train/test split for the new mask dataset.
3. Retrain MK-UNet (binary) on Modal, with fixed train/eval code.
4. Site-holdout evaluation of the segmentation model.
5. Connected-component counting logic + count-accuracy evaluation.
6. Geometric pattern-classification rule + manual-agreement evaluation.
7. Wire into `pipeline.py`, replacing the ResNet-18 Stage 2/3 calls.
