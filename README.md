# Flooded Road and Building Mapping from Satellite Imagery using Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![SpaceNet 8](https://img.shields.io/badge/Dataset-SpaceNet%208-blue.svg)](https://spacenet.ai/sn8-challenge/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Detection of buildings and roads, and of which ones are flooded, from co-registered
pre-/post-event satellite imagery (SpaceNet 8), with a web interface for inspecting
predictions against the ground truth.

---

## Highlights

- **Co-registration (O1)**: post-event tiles are resampled onto the pre-event pixel grid from
  their GeoTIFF georeferencing, then the residual offset is estimated (block-wise phase
  correlation) and corrected.
- **Baseline (O2)**: ResNet34 U-Net on early-concatenated pre/post imagery (6 channels).
- **Siamese cross-attention fusion (O3)**: shared-weight ResNet34 encoders with bidirectional
  cross-attention and a gated change branch at every scale.
- **Object-level outputs (O4)**: building polygons and a road graph, each flood-attributed by
  majority vote (SpaceNet 8 baseline convention), with counts, road length and flooded share.
- **Web interface**: pick any tile from the dataset (its post image is matched automatically
  from the mapping CSV) or upload a GeoTIFF pair; compare before/after and overlay ground truth
  with per-tile IoU/F1.
- **Testing & deployment (O5)**: pytest suite, CUDA Docker image that also builds the frontend.
- **Research pipeline**: RA-Fusion (frozen DINOv3 satellite ViT-L/16 + learned coarse-to-fine
  pre/post alignment + change-gated fusion) against the SpaceNet 8 baseline, a Siamese U-Net
  (5th-place style) and SNUNet-CD, under three geographic protocols with APLS, instance F1 and
  bootstrap confidence intervals; trained automatically on Kaggle from GitHub Actions.

### Dataset finding: residual pre/post misalignment

Even after georeferenced resampling, SpaceNet 8 pairs remain offset
([`docs/analysis/alignment_stats.csv`](docs/analysis/alignment_stats.csv),
`scripts/analyze_alignment.py`):

| AOI | Pairs (confident estimate) | Median offset | 90th percentile | > 8 px |
|---|---|---|---|---|
| Germany | 282 (218) | 12.1 px / 4.1 m | 43.1 px / 14.7 m | 58 % |
| Louisiana-East | 599 (527) | 11.6 px / 4.3 m | 35.7 px / 14.2 m | 78 % |

The offset is mostly a rigid shift (median within-tile spread ≈ 1.5 px). In Germany the choice
of post image matters: the 0.25 m post image is offset by 4.3 px (median) versus 23 px for the
0.57 m one, so the pair index orders post images best-aligned first.

---

## Project Structure

```
CAPSTONE-PROJECT/
├── backend/                   FastAPI service
│   ├── main.py                App, CORS, model loading, serves frontend/dist at /
│   └── routes.py              /api/health, /api/pairs*, /api/predict, /api/predict_pair
├── frontend/                  React (Vite) interface — src/App.jsx, src/index.css
├── dataloader/                Data pipeline
│   ├── align.py               Post→pre resampling, residual shift estimation/correction
│   ├── pairs.py               Pre/post pair index from *_label_image_mapping.csv
│   ├── ground_truth.py        Reference buildings/roads in pixel space + raster mask
│   ├── mask_generator.py      5-channel masks from GeoJSON + reference CSV
│   ├── dataset.py             PyTorch Dataset, splits, augmentations
│   ├── crs.py, tiling.py      CRS reprojection and tiling utilities
│   └── prepare.py             Full-tile training data: best-aligned post, targets, folds/splits
├── models/
│   ├── unet_baseline.py       O2 baseline (served by the web interface)
│   ├── siamese_fusion.py      O3 Siamese cross-attention fusion (served by the web interface)
│   ├── ra_fusion.py           RA-Fusion research model
│   ├── backbones.py           Frozen DINOv3 satellite ViT-L/16, timm CNN backbones
│   ├── alignment.py           Correlation-based coarse-to-fine flow alignment
│   ├── baselines/             SN8 baseline, Siamese U-Net, SNUNet-CD
│   └── legacy.py              Adapter running the v0 models in the research pipeline
├── training/
│   ├── engine.py              Training loop: AMP, EMA, stages, per-epoch resumable checkpoints
│   ├── dataset.py             Random crops, synthetic misalignment, self-supervised pairs
│   ├── losses.py              Multi-task loss (object-masked flood, alignment)
│   ├── inference.py           Sliding-window prediction with flip TTA
│   ├── evaluate_run.py        Threshold tuning, test metrics, misalignment stress test, timing
│   └── queue.py, config.py    Experiment queue over experiments/runs.yaml
├── evaluation/                APLS, building instance F1, flooded-road length F1, bootstrap CIs
├── experiments/runs.yaml      Every run of the study (models × protocols × seeds, ablations)
├── orchestration/             Kaggle session script and the GitHub Actions orchestrator
├── inference/
│   ├── predict.py             FloodPredictor: aligned loading, sliding window, thresholds
│   ├── postprocess.py         Building polygons, road graph, flood attribution, summary
│   └── eval.py                Per-tile pixel metrics, RGBA overlay
├── scripts/                   prepare_data.py, run_queue.py, gpu_probe.py, aggregate_results.py,
│                              analyze_alignment.py
├── legacy/                    v0 training pipeline (Colab), kept to reproduce the earlier weights
├── tests/                     pytest suite
├── docs/
│   ├── analysis/              Dataset analyses (alignment_stats.csv)
│   └── training-logs/         Colab training logs
├── weights/                   Model checkpoints (*.pth, not in git) + metrics
├── .github/workflows/         train.yml — Kaggle training automation
├── Dockerfile, docker-compose.yml, .dockerignore
└── requirements.txt
```

`CP1-DATASET/` (SpaceNet 8 AOIs, not in git) is expected at the project root:

```
CP1-DATASET/
├── Germany_Training_Public/        PRE-event/, POST-event/, annotations/, *_label_image_mapping.csv, *_reference.csv
├── Louisiana-East_Training_Public/ (same layout)
└── Louisiana-West_Test_Public/     PRE-event/, POST-event/, mapping CSV (no public labels)
```

---

## Setup

### Weights

Download `fusion_net_best.pth` and `unet_net_best.pth` from the
[Google Drive weights folder](https://drive.google.com/drive/folders/1FYg7WhwLALRmfS8VdveY_0WzItsQ2L22?usp=sharing)
into `weights/`. The current files are from the final Colab run (fusion: epoch 48, val mIoU
0.178; U-Net: epoch 26, val mIoU 0.175).

### Install

```bash
git clone https://github.com/isuryaprakashh/Capstone-Flood-Mapping.git
cd Capstone-Flood-Mapping
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Data

Download the SpaceNet 8 tarballs into `CP1-DATASET/` (the archives have no top-level AOI
folder, so each is extracted into its own):

```bash
for aoi in Germany_Training_Public Louisiana-East_Training_Public Louisiana-West_Test_Public; do
  curl -O https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/$aoi.tar.gz
  mkdir -p CP1-DATASET/$aoi && tar -xzf $aoi.tar.gz -C CP1-DATASET/$aoi
done
```

---

## Running the Web Interface

Settings are read from `.env` (`USE_STUB`, `MODEL_TYPE`, `O3_FUSION_WEIGHTS`,
`O2_BASELINE_WEIGHTS`, `SPACENET8_DATA_DIR`).

```bash
# Terminal 1 — API (loads weights/fusion_net_best.pth unless USE_STUB=true)
python backend/main.py

# Terminal 2 — frontend dev server (http://localhost:3000, proxies /api → :8000)
cd frontend && npm run dev
```

Or build once and let the API serve it: `cd frontend && npm run build`, then open
`http://localhost:8000`. API docs: `http://localhost:8000/api/docs`.

**Usage:** choose an AOI and search a tile (e.g. `1_15_82`) — the post image is selected
automatically — then *Map flood damage*. For labelled AOIs, enable *Ground truth outlines* to
compare and read the per-tile IoU/F1. In *Upload files* mode, choosing a pre-event GeoTIFF that
appears in a mapping CSV attaches its post image automatically.

### Docker

```bash
docker compose up --build   # http://localhost:8000 — mounts ./weights and ./CP1-DATASET
```

---

## Training & Evaluation

### Protocols

| Protocol | Train | Validation | Test |
|---|---|---|---|
| P1 | Louisiana-East (479) | Louisiana-East (120) | Germany (202) |
| P2 | Germany (181) | Germany (21) | Louisiana-East (599) |
| P3 | pooled, spatially blocked folds (494) | 166 | 141 |

Louisiana-West has no public labels, so it is not used for evaluation.

### Local

```bash
# Full 1300 px tiles with the best-aligned post image, targets and splits (~3.5 GB)
python scripts/prepare_data.py --data-root ./CP1-DATASET --out-dir ./processed

# Every run in experiments/runs.yaml; resumes paused or failed runs
python scripts/run_queue.py --processed ./processed --results ./results

# One short run per model, including a stop/resume check
python scripts/run_queue.py --processed ./processed --results ./results_smoke --smoke
```

Each run writes `results/<run>/`: `train_log.csv`, `metrics_val.json` (tuned thresholds),
`metrics_test.json` (metrics with 95 % bootstrap CIs), `stress_test.json` (metrics under added
pre/post offsets), `timing.json` and `checkpoints/best.pt`. `results/PROGRESS.md` summarises all
runs. `python scripts/aggregate_results.py --results ./results` builds the tables
(`RESULTS.md`, `RESULTS.csv`: mean ± std over seeds, misalignment stress curves).

### Automated (Kaggle + GitHub Actions)

[`.github/workflows/train.yml`](.github/workflows/train.yml) runs every 30 minutes and drives
Kaggle T4×2 sessions: a smoke test of every model, a CPU dataset-preparation kernel, then training
sessions that each resume from the previous session's checkpoints. Progress (`STATUS.md`,
`PROGRESS.md`, per-run logs and metrics) is committed to `experiments/results/` on the `results`
branch, which merges into `main` without conflicts; checkpoints stay in the Kaggle kernel outputs.

Setup: repository secret `KAGGLE_API_TOKEN` and repository variable `KAGGLE_USERNAME`. Manual
actions (*Actions → Train on Kaggle → Run workflow*): `tick`, `update-code` (re-pin training to
the current `main`), `resume` (after a stop).

### Legacy pipeline

The v0 U-Net and Siamese models in `weights/` were trained with the Colab pipeline in
[`legacy/`](legacy/README.md).

Re-run the misalignment analysis:

```bash
python scripts/analyze_alignment.py --data-root ./CP1-DATASET --output docs/analysis/alignment_stats.csv
```

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Mask Channels

| Channel | Class | Source |
|---|---|---|
| 0 | `building` | GeoJSON building polygons |
| 1 | `road` | Reference CSV road centerlines (buffered) |
| 2 | `flood` | Union of channels 3 and 4 |
| 3 | `flooded_building` | Buildings attributed flooded |
| 4 | `flooded_road` | Roads attributed flooded |

---

## Acknowledgements

SpaceNet 8 dataset: R. Hänsch et al., *SpaceNet 8 — The Detection of Flooded Roads and
Buildings*, CVPR Workshops 2022. Imagery © Maxar Technologies (CC BY-NC 4.0).

RA-Fusion uses DINOv3 (O. Siméoni et al., *DINOv3*, 2025; satellite ViT-L/16 weights via
[timm](https://github.com/huggingface/pytorch-image-models)), distributed under the DINOv3
License.

KL University Capstone Project KLCAP-2026-00010 — Department of Computer Science & Engineering.
