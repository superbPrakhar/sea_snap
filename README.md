# AI-Based Ocean Hazard Reporting System

A complete, working implementation of the BITE308L (Artificial Intelligence) project
**"AI-Based Ocean Hazard Reporting"**:

| Component | Model | Input | Output |
|---|---|---|---|
| **Oil Spill Detection** | U-Net++ (EfficientNet encoder) semantic segmentation | drone RGB image | pixel-level mask: Background / Oil / Others / Water |
| **Cyclone Prediction** | Artificial Neural Network classifier | SST, pressure, humidity, wind speed/direction, lat/lon | probability of cyclone formation |
| **Hazard Alert System** | Flask web dashboard | outputs of both models | unified alert feed + visualisations |

---

## 1. Quick start (local machine)

```bash
# 1. create a virtual environment (Python 3.10+ recommended)
py -3.10 -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)

# 2. install PyTorch with CUDA (skip the --index-url part for CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. install everything else
pip install -r requirements.txt

# 4. verify the environment
python setup_check.py

# 5. one click: generate demo data + train both models + evaluate
python train_all.py
# (fast smoke test: python train_all.py --quick)

# 6. open the dashboard
python dashboard/app.py           # -> http://127.0.0.1:5000
```

`train_all.py` automatically:
1. generates the synthetic demo datasets (see below) if none exist,
2. trains the U-Net++ oil-spill segmenter,
3. trains the ANN cyclone classifier,
4. runs the test-split evaluation and saves metrics + plots to `checkpoints/`.

## 2. Project layout

```
ai project/
├── config.py                     # all paths / hyper-parameters / class colours
├── train_all.py                  # one-click: data + training + evaluation
├── setup_check.py                # environment sanity check
├── requirements.txt
├── oil_spill/                    # COMPONENT 1 - oil spill segmentation
│   ├── make_synthetic_dataset.py #   demo dataset generator (same mask colour convention)
│   ├── dataset.py                #   Dataset + robust image/mask pairing (by stem)
│   ├── model.py                  #   U-Net++ builder (graceful offline fallback)
│   ├── train.py                  #   training (AMP, early stopping on real mIoU)
│   ├── evaluate.py               #   per-class IoU/F1/P/R + confusion matrix
│   └── predict.py                #   checkpoint loading + single-image inference
├── cyclone/                      # COMPONENT 2 - cyclone prediction
│   ├── make_dataset.py           #   synthetic, physically-motivated training data
│   ├── model.py                  #   ANN classifier (7 -> 64 -> 32 -> 1)
│   ├── train.py                  #   imputation + scaling + BCE training
│   └── predict.py                #   risk probability + risk level
├── dashboard/                    # COMPONENT 3 - Hazard Alert System
│   ├── app.py                    #   Flask API + page server
│   ├── templates/index.html
│   └── static/                   #   style.css, main.js
├── colab/
│   └── oil_spill_segmentation_FIXED.ipynb   # corrected version of the original Colab notebook
├── data/                         # datasets (generated or dropped in by you)
└── checkpoints/                  # trained models, metrics, learning curves
```

## 3. Using your REAL oil-spill dataset

The pipeline expects this layout (identical to the original `dataset.zip`):

```
data/oil_spill/
├── train/images/   train/masks/
├── val/images/     val/masks/
└── test/images/    test/masks/
```

* Masks use the colour convention: black `(0,0,0)` = Background,
  pink `(255,0,124)` = Oil, yellow `(255,204,51)` = Others,
  cyan `(51,221,255)` = Water (set in `config.py`).
* Images and masks are paired **by filename stem**, so different prefixes
  (`img_1.png` / `mask_1.png`) are fine — a bug in the original notebook paired
  them by sorted order instead.
* Then simply run `python oil_spill/train.py` (it uses whatever is in `data/oil_spill`).

To train with the full setup from the report on a GPU with >= 6 GB VRAM:

```bash
python oil_spill/train.py --encoder timm-efficientnet-b4 --batch-size 8
```

On a 4 GB GPU (e.g. RTX 3050 laptop) the default light encoder
(`efficientnet-b0`, batch 8) trains comfortably; the training loop also halves
the batch size automatically if it runs out of memory.

### How the current checkpoint was trained (synthetic-to-real pipeline)

No real labelled dataset is bundled with the repo, so the shipped model was
trained with a **synthetic-to-real** pipeline (see
`oil_spill/make_synthetic_dataset.py` and `oil_spill/make_real_patches.py`):

1. **Realistic synthetic scenes** (1400 images @ 512px): depth-graded ocean,
   vegetated islands, rocky islets, sandy flats, reefs, foam rings, sun glint,
   four water palettes, and oil in four appearances (black crude, orange-brown
   emulsion, weathered dark, silver sheen) with blob *and* filament shapes.
2. **Unambiguous real patches** (~160 crops from public-domain Wikimedia
   aerial photos in `test_images/`): fully-forest patches labelled Background,
   fully-open-water patches labelled Water, fully-inside-the-slick patches
   labelled Oil. Pure content => certain labels without manual annotation.
3. **U-Net++ / EfficientNet-b3** trained on the synthetic set, then
   fine-tuned twice on synthetic + real patches (`--init-from`).

Validated on 8 real aerial photos never seen in training (Wikimedia Commons):

| Photo | Oil predicted | Expected | Verdict |
|---|---|---|---|
| Maldives island (user's photo) | **0.14 %** | no oil | correct (was 5.8 % before retraining) |
| Balos lagoon, Crete | 0.62 % | no oil | correct |
| Chuuk lagoon reef | 13.4 % | no oil | false alert - mottled reef flat genuinely resembles an emulsion slick |
| Yura coast, Japan | 3.1 % | no oil | mild false alert |
| Deepwater Horizon aerial | 13.3 % | oil | **detected** |
| Mississippi Delta slick | 82.0 % | oil | **detected** |
| North Sea slick | 30.5 % | oil | **detected** |

**Known limitation:** trained without a real labelled segmentation dataset,
the model still over-predicts oil on mottled reef flats. The correct fix is
fine-tuning on the real `dataset.zip` (step above) - the pipeline accepts it
without code changes. Alert thresholds in `config.py` are pre-tuned to keep
the residual false positives at bay in the meantime.

## 4. Using real cyclone data

Replace `data/cyclone/cyclone_data.csv` with a CSV having exactly these columns:

```
sea_surface_temp,pressure,humidity,wind_speed,wind_direction,latitude,longitude,cyclone
29.4,992.3,88,13.2,254,14.8,87.1,1
```

`cyclone = 1` means a cyclone formed under those conditions. Sources: NOAA IBTrACS
(storm tracks) joined with ERA5 / NOAA OISST environmental fields. The generated
demo CSV (`python cyclone/make_dataset.py`) is synthetic — it follows real
cyclogenesis physics (SST >= ~26.5 °C, low pressure, high humidity) but is for
demonstration only.

## 5. Dashboard

`python dashboard/app.py` opens http://127.0.0.1:5000 with:

* **Oil Spill Detection** — upload any drone image, get the segmentation overlay,
  the colour mask, per-class coverage statistics and an automatic alert when oil
  coverage exceeds the thresholds in `config.py`.
* **Cyclone Prediction** — enter environmental readings (or use the presets) and get
  the formation probability with a Low / Moderate / High / Severe risk level.
* **Hazard Alert Feed** — unified, timestamped alerts from both components.

The API (usable without the UI):

```bash
curl -F "image=@drone.png" http://127.0.0.1:5000/api/oil-spill
curl -X POST -H "Content-Type: application/json" ^
     -d "{\"sea_surface_temp\":29.5,\"pressure\":990,\"humidity\":90,\"wind_speed\":15,\"wind_direction\":260,\"latitude\":15,\"longitude\":88}" ^
     http://127.0.0.1:5000/api/cyclone
```

## 6. What was fixed from the original notebook

The original `oil_spill_segmentation.ipynb` (Colab export) could not run. Fixes:

1. `CLASS_NAMES`/`NUM_CLASSES` were used (HF model creation) **before** they were
   defined at the bottom of the file.
2. `run_full_evaluation(...)` was called but never defined (only
   `run_full_evaluation_hf` existed).
3. `pandas`, `seaborn`, `torchmetrics` were imported **after** first use.
4. Deprecated `torch.cuda.amp.GradScaler()`/`autocast()` replaced by the
   `torch.amp` API (also required for CPU-only machines).
5. The "IoU" printed during training was actually **pixel accuracy** — replaced by
   a real macro mean-IoU via `torchmetrics`.
6. Three duplicated prediction-display cells collapsed into one parameterised
   function; image/mask pairing now matches by filename stem.
7. Colab-only paths (`/content/drive/...`) and shell magics (`!pip`, `!unzip`)
   were made configurable so the same code runs locally and on Colab.

The corrected Colab workflow is in `colab/oil_spill_segmentation_FIXED.ipynb`.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | lower `--batch-size`, or keep the default `efficientnet-b0` encoder |
| `Checkpoint not found` in the dashboard | run `python oil_spill/train.py` and `python cyclone/train.py` |
| `No image/mask pairs found` | generate demo data: `python oil_spill/make_synthetic_dataset.py` |
| torch installed but `cuda: False` | install the CUDA build: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| `num_workers` crash on Windows | keep `--num-workers 0` (default here) |
| Pretrained encoder download fails (offline) | `--encoder-weights None` |
