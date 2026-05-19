# Facial Emotion Recognition (FER2013) — Streamlit App

Section 10.3 homework: a deep-learning emotion recogniser trained on FER2013, with
a Streamlit UI for training, evaluation, and inference. Includes three CNN
architectures, a Scikit-Learn baseline, full checkpointing, and a "stylized / NPR"
inference path for non-photorealistic character art.

## Features

| Spec requirement | Implementation |
|---|---|
| Streamlit UI: upload, start/pause/resume/stop, log, output area | `app.py` |
| ≥3 CNN architectures | `SimpleCNN`, `DeeperCNN`, `MiniResNet` in `src/models.py` |
| Multi-framework support | **PyTorch** (CNNs) + **Scikit-Learn** (logreg baseline) |
| Hyperparameter selection | Sidebar: optimizer, LR, weight decay, activation, dropout, batch size, epochs, augmentation, scheduler |
| Train-from-scratch & Resume | "Resume from" dropdown |
| Checkpoints: best / last / on-pause / on-stop | `src/trainer.py` |
| Live training metrics + log | Train tab |
| Accuracy + confusion matrix + ROC curves | Evaluate tab |
| FER2013 preprocessing | Grayscale + resize 48×48 + normalise + RandomFlip/Affine/Erasing |

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get the data (~60 MB)
python scripts/download_fer2013.py

# 3. Run the UI
streamlit run app.py
```

Open http://localhost:8501 — you'll see four tabs: **Train**, **Evaluate**,
**Predict**, **About**.

## Dataset layouts supported

The dataset auto-loader in `src/dataset.py` handles both common formats:

1. **CSV format** — `data/fer2013.csv` with columns `emotion, pixels, Usage`.
2. **ImageFolder format** — `data/{train,test}/<emotion>/*.jpg` (the Kaggle layout).
   In this case a 10% validation split is carved out of `train/` deterministically.

If `download_fer2013.py` fails (the HuggingFace mirror is occasionally rate-limited),
download manually from <https://www.kaggle.com/datasets/msambare/fer2013> and
unzip the result into `./data/`.

## Repo layout

```
app.py                       # Streamlit UI
scripts/download_fer2013.py  # Dataset downloader
src/
├── models.py                # 3 CNN architectures + activation registry
├── dataset.py               # FER2013 loader (CSV + ImageFolder), transforms
├── trainer.py               # Background-thread Trainer with pause/resume/stop
├── evaluate.py              # Confusion matrix, ROC, per-class report
├── predict.py               # Inference + NPR-friendly preprocessing
└── sklearn_baseline.py      # Logistic-regression baseline
data/                        # gitignored — dataset goes here
checkpoints/                 # gitignored — model weights go here
REPORT.md                    # Experimental report (Section 10.3.4)
```

## Inference on stylised / NPR character art

FER2013 is photographic 48×48 grayscale faces. Cel-shaded or NPR character
renders look very different — flat shading, hard line art, exaggerated
features. The Predict tab has a **"Stylized / NPR mode"** toggle that applies
histogram equalisation and a small Gaussian blur before the standard FER2013
transforms. This usually produces sensible readings on toon-shaded characters,
though for best results you should fine-tune the model on a stylised face
dataset (the homework only requires FER2013).

## Tips

- On CPU, `SimpleCNN` finishes 15 epochs in a few minutes. `MiniResNet`
  benefits massively from a GPU.
- The training thread checkpoints automatically: every epoch (`*_last.pth`),
  on a new best validation accuracy (`*_best.pth`), on pause (`*_pause.pth`),
  and on stop (`*_stopped.pth`).
- The **Evaluate** tab's ROC curves and confusion matrix are exactly what the
  homework report needs — screenshot them or copy into `REPORT.md`.
