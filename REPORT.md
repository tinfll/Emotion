# Experimental Report: Facial Emotion Recognition on FER2013

> Skeleton report aligned with Section 10.3.4. Numbers in tables are placeholders
> labelled `<fill in>`; replace them with values produced by the Evaluate tab
> after you actually train the three models.

---

## 1. Project Objectives and Requirements Analysis

The goal of this project is to build an end-to-end facial emotion recognition
system based on the FER2013 dataset, exposed through a Streamlit user
interface. The system must support:

- **Image input** through a file uploader.
- **Multiple CNN architectures** that the user can switch between.
- **Hyperparameter configuration** at runtime (learning rate, optimizer,
  activation, dropout, batch size, augmentation, LR scheduler).
- **Training control** with start / pause / resume / stop semantics, running on
  a background thread so the UI remains responsive.
- **Multiple checkpointing strategies**: best validation accuracy, last epoch,
  on-pause auto-save, on-stop auto-save.
- **Quantitative evaluation** with accuracy, per-class precision/recall/F1,
  confusion matrix, and one-vs-rest ROC curves.
- **Inference** on user-supplied images, including a stylised / NPR mode for
  cel-shaded character art that lies outside the FER2013 distribution.

The codebase is organised in `src/` with one module per concern (models,
dataset, trainer, evaluation, prediction, sklearn baseline) plus the Streamlit
app (`app.py`). The CNNs use PyTorch; a Scikit-Learn logistic-regression
baseline is included to satisfy the "supports multiple frameworks" requirement
and to give us a non-deep-learning reference point.

## 2. Data Preprocessing

FER2013 contains 35,887 grayscale 48×48 images across 7 emotion classes
(`Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`). Standard splits:

| Split | Count |
|---|---|
| Train | 28,709 |
| Validation | 3,589 |
| Test | 3,589 |

**Pre-processing pipeline** (`src/dataset.py`):

1. Force grayscale with one channel.
2. Resize to 48×48 (no-op for the canonical CSV; needed for ImageFolder JPGs).
3. Convert to tensor (range `[0, 1]`).
4. Normalise with mean=0.5077, std=0.2550 (precomputed on the FER2013 train
   split).

**Training-time augmentation** (`augment=True`):

- `RandomHorizontalFlip` — facial expressions are bilaterally symmetric, so
  horizontal flips double the effective data without distorting labels.
- `RandomAffine(degrees=10, translate=5%, scale±5%)` — robustness to small
  pose / framing variations.
- `RandomErasing(p=0.2, scale=2–10%)` — discourages over-reliance on small
  local features (cf. Cutout / Random Erasing literature).

**Class imbalance**: `Disgust` is heavily under-represented (~547 training
samples). The current configuration uses cross-entropy with label smoothing
(`label_smoothing=0.05`) to gently regularise; per-class accuracy on `Disgust`
is therefore the metric to watch. A re-weighted loss is a natural follow-up.

## 3. CNN Model Designs and Parameter Settings

Three architectures are implemented in `src/models.py`:

### Model A — SimpleCNN (~1.2M params)

```
Conv(1→32, 3×3) → ReLU → MaxPool
Conv(32→64, 3×3) → ReLU → MaxPool
Flatten → Dropout → Linear(64·12·12 → 128) → ReLU → Dropout → Linear(128 → 7)
```

A shallow baseline. Trains in ~30s/epoch on CPU. Establishes a lower bound
above the chance rate (~14%).

### Model B — DeeperCNN (~3.5M params)

VGG-style stack of three `(Conv-BN-Act-Conv-BN-Act-MaxPool)` blocks at widths
64 / 128 / 256, followed by `Dropout-Linear(256·6·6→256)-Act-Dropout-Linear(256→7)`.
BatchNorm stabilises training at higher learning rates and Dropout regularises
the dense head. This is the default choice in the UI.

### Model C — MiniResNet (~2.8M params)

A ResNet-style network: a 3×3 stem at width 64, then three stages of two
residual blocks each, doubling channels (64 → 128 → 256) and halving spatial
resolution at each stage transition. Ends with global average pooling and a
single 256→7 linear head. Skip connections give it the best gradient flow of
the three.

### Default hyperparameters

| Param | Default | Rationale |
|---|---|---|
| Optimizer | Adam | Robust default for image classification at this scale |
| Learning rate | 1e-3 | Adam-friendly; cosine-decayed to 0 over `epochs` |
| Weight decay | 1e-4 | Mild regularisation |
| Dropout | 0.4 | Noticeable but not crippling on FER2013 |
| Batch size | 64 | Fits on consumer GPUs and CPUs |
| Epochs | 15 | Enough for clear convergence trends; bump to 40+ for best results |
| Augmentation | on | See preprocessing section |
| Scheduler | cosine | Smooth annealing, no manual step tuning |

## 4. Experimental Results

> Run all three models from the Train tab, then run the Evaluate tab on each
> `*_best.pth` checkpoint. Fill in the table below from the metrics it shows.

### 4.1 Headline numbers (test set)

| Model | Params | Test accuracy | Macro AUC | Best epoch |
|---|---:|---:|---:|---:|
| SimpleCNN  | 1.2M  | `<fill in>` | `<fill in>` | `<fill in>` |
| DeeperCNN  | 3.5M  | `<fill in>` | `<fill in>` | `<fill in>` |
| MiniResNet | 2.8M  | `<fill in>` | `<fill in>` | `<fill in>` |
| LogReg (sklearn baseline) | 17k+ | `<fill in>` | n/a | — |

For reference, FER2013 has a public leaderboard ceiling around 73-76% accuracy
for ensembles; single-model results in the 65-70% range are typical and
expected for this homework.

### 4.2 Per-class breakdown

The Evaluate tab produces a classification report. Paste the table for the
best-performing model here. Expected pattern:

- `Happy` and `Surprise` are the easiest classes (distinctive facial features).
- `Disgust` is the hardest (under-represented).
- `Fear` ↔ `Sad` and `Angry` ↔ `Disgust` are the most common confusions.

### 4.3 Confusion matrices

Insert the row-normalised confusion matrices for each model side by side. The
diagonal density visually summarises overall accuracy; off-diagonal "hot
zones" reveal which emotions the model conflates.

### 4.4 ROC curves

Insert the one-vs-rest ROC plot for the best model. Each emotion's curve and
its AUC are shown in the legend. Look for:

- Tight curves toward the top-left for `Happy` / `Surprise`.
- Lower AUC on `Disgust` and `Fear`, mirroring the imbalance and visual
  ambiguity of those classes.

### 4.5 Comparison and analysis

Expected qualitative findings (replace with what your run actually shows):

- **Depth helps but with diminishing returns.** Going SimpleCNN → DeeperCNN
  typically adds 8-15 accuracy points; DeeperCNN → MiniResNet adds another
  1-3.
- **Skip connections matter more at depth.** MiniResNet's residuals let it
  train deeper without vanishing-gradient issues, but on a 48×48 input the
  effective depth is small so the win over DeeperCNN is modest.
- **Augmentation is the single highest-leverage knob** at this scale; turning
  it off costs more than halving the model size.
- **Logistic regression on raw pixels** sits well below the CNNs (often around
  35-40%), confirming that the spatial structure CNNs exploit is what makes
  the task tractable.

## 5. Challenges and Solutions

1. **Pause / resume in Streamlit.** Streamlit re-runs the entire script on
   every interaction. Solution: run training in a `threading.Thread`, store
   the `Trainer` instance in `st.session_state`, and use `threading.Event`s
   for pause/stop signalling. The UI thread reads thread-safe snapshots of
   the trainer state and re-renders charts via `st.rerun()` while training is
   alive.

2. **Auto-save on pause / stop.** The `_wait_if_paused` and `_run`-end paths
   call `save_checkpoint("pause")` / `save_checkpoint("stopped")` so no
   training progress is ever lost.

3. **Class imbalance for `Disgust`.** Mitigated for now with label smoothing.
   A weighted cross-entropy loss (`weight=1/freq`) is a small follow-up.

4. **Stylised / NPR character art.** FER2013 is photographic; cel-shaded
   characters fall outside its distribution. Adding histogram equalisation
   and a small Gaussian blur before the standard transforms (`stylized=True`
   in `src/predict.py`) makes the model usable on this out-of-distribution
   input. The proper fix is fine-tuning on a stylised dataset.

5. **Sklearn API drift.** `LogisticRegression(multi_class="multinomial")` is
   deprecated in scikit-learn ≥1.5; the default with the `lbfgs` solver is
   already multinomial, so the parameter was removed.

6. **Dataset format variance.** FER2013 ships in two flavours (canonical CSV
   vs. Kaggle ImageFolder). The loader auto-detects which is present and
   carves a 10% val split deterministically when only train/test exist.

## 6. Summary and Improvements

The system meets every functional requirement of Section 10.3.2 / 10.3.3 and
gives the user enough hyperparameter control to reproduce the comparative
study required for the report. Suggested improvements:

- **Class-weighted loss** to lift `Disgust` recall.
- **Test-time augmentation** (10-crop / horizontal-flip averaging) typically
  buys ~1% accuracy at no training cost.
- **Pretrained-backbone transfer learning** — even a small grayscale-adapted
  ResNet-18 pretrained on ImageNet outperforms training from scratch.
- **Stylised fine-tuning dataset** for the NPR / OC character use case:
  collect ~1k labelled toon-shaded faces and fine-tune for 2-3 epochs from
  the FER2013 best checkpoint.
- **Mixed precision training** (`torch.amp`) for faster GPU iterations.
- **Per-emotion threshold tuning** based on the ROC curves to optimise
  application-level metrics rather than top-1 accuracy.
