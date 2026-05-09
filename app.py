"""Streamlit UI for the FER2013 Facial Emotion Recognition project.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.dataset import EMOTION_LABELS, build_dataloaders, class_distribution
from src.evaluate import collect_predictions, compute_metrics, plot_confusion_matrix, plot_roc_curves
from src.models import MODEL_REGISTRY, build_model, count_parameters
from src.predict import predict_image
from src.trainer import OPTIMIZERS, TrainConfig, Trainer, list_checkpoints
from src.models import ACTIVATIONS


st.set_page_config(page_title="FER2013 Emotion Recognition", page_icon=":blush:", layout="wide")

DATA_DIR_DEFAULT = "data"
CKPT_DIR_DEFAULT = "checkpoints"

# ---------------------------------------------------------------- session state

ss = st.session_state
ss.setdefault("trainer", None)
ss.setdefault("data_dir", DATA_DIR_DEFAULT)
ss.setdefault("ckpt_dir", CKPT_DIR_DEFAULT)
ss.setdefault("loaders", None)
ss.setdefault("loader_key", None)


def get_dataloaders(data_dir: str, batch_size: int, augment: bool):
    key = (os.path.abspath(data_dir), batch_size, augment)
    if ss.loader_key != key:
        ss.loaders = build_dataloaders(data_dir, batch_size=batch_size, augment=augment, num_workers=0)
        ss.loader_key = key
    return ss.loaders


# --------------------------------------------------------------------- sidebar

with st.sidebar:
    st.title(":blush: FER2013")
    st.caption("Facial Emotion Recognition — homework UI")

    st.subheader("1. Framework")
    framework = st.radio(
        "Backend",
        ["PyTorch (CNNs)", "Scikit-Learn (logreg baseline)"],
        help="The CNN models use PyTorch. A Sklearn logistic-regression baseline "
             "is provided so you can compare deep vs. classical methods.",
    )

    st.subheader("2. Data")
    ss.data_dir = st.text_input("Dataset directory", ss.data_dir,
                                help="Run `python scripts/download_fer2013.py` to populate this.")
    ss.ckpt_dir = st.text_input("Checkpoints directory", ss.ckpt_dir)

    if st.button("Inspect dataset", use_container_width=True):
        try:
            dist = class_distribution(ss.data_dir)
            st.success(f"Found {sum(dist.values())} training images")
            st.bar_chart(pd.Series(dist, name="count"))
        except Exception as e:
            st.error(f"{e}")

    st.subheader("3. Model")
    model_name = st.selectbox("Architecture", list(MODEL_REGISTRY.keys()), index=1)
    st.caption(MODEL_REGISTRY[model_name]["desc"])

    st.subheader("4. Hyperparameters")
    optimizer = st.selectbox("Optimizer", list(OPTIMIZERS.keys()), index=0)
    activation = st.selectbox("Activation", list(ACTIVATIONS.keys()), index=0)
    lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 1e-2], value=1e-3,
                          format_func=lambda v: f"{v:.0e}")
    weight_decay = st.select_slider("Weight decay", options=[0.0, 1e-5, 1e-4, 1e-3], value=1e-4,
                                    format_func=lambda v: f"{v:.0e}")
    dropout = st.slider("Dropout", 0.0, 0.6, 0.4, 0.05)
    batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=64)
    epochs = st.slider("Epochs", 1, 60, 15)
    augment = st.checkbox("Data augmentation", True)
    scheduler = st.selectbox("LR scheduler", ["cosine", "step", "none"], index=0)


# ---------------------------------------------------------- main: tabs

st.title("Facial Emotion Recognition (FER2013)")
st.caption("Section 10.3 homework — Streamlit UI · 3 CNN architectures · checkpointing · "
           "confusion matrix · ROC curves · NPR-image inference mode.")

tabs = st.tabs([":rocket: Train", ":bar_chart: Evaluate", ":mag: Predict", ":books: About"])


# =========================================================== Train tab

with tabs[0]:
    st.subheader("Training control")

    if framework.startswith("Scikit"):
        st.info("Sklearn baseline trains in one shot (no live curve). Click below to fit a "
                "logistic-regression model on raw 48×48 pixels.")
        if st.button(":sparkles: Fit Sklearn baseline"):
            from src.sklearn_baseline import train_sklearn_baseline
            with st.spinner("Fitting logistic regression…"):
                try:
                    res = train_sklearn_baseline(ss.data_dir, save_path=str(Path(ss.ckpt_dir) / "sklearn_logreg.pkl"))
                    st.success(f"val_acc={res['val_acc']:.4f}  test_acc={res['test_acc']:.4f}")
                    st.caption(f"Saved to {res['save_path']}")
                except Exception as e:
                    st.error(f"{e}")
    else:
        cfg = TrainConfig(
            model_name=model_name, optimizer=optimizer, learning_rate=lr,
            weight_decay=weight_decay, activation=activation, dropout=dropout,
            batch_size=batch_size, epochs=epochs, augment=augment, scheduler=scheduler,
        )

        c1, c2, c3, c4 = st.columns(4)
        existing_ckpts = list_checkpoints(ss.ckpt_dir)
        resume_path = c4.selectbox("Resume from", ["(train from scratch)"] + existing_ckpts, index=0)

        start_clicked = c1.button(":arrow_forward: Start", type="primary", use_container_width=True,
                                   disabled=(ss.trainer is not None and ss.trainer.is_alive()))
        pause_resume_label = ":arrow_forward_double: Resume" if (ss.trainer and ss.trainer.snapshot()["status"] == "paused") else ":pause_button: Pause"
        pause_clicked = c2.button(pause_resume_label, use_container_width=True,
                                   disabled=(ss.trainer is None or not ss.trainer.is_alive()))
        stop_clicked = c3.button(":stop_button: Stop", use_container_width=True,
                                  disabled=(ss.trainer is None or not ss.trainer.is_alive()))

        if start_clicked:
            try:
                tr_loader, val_loader, _ = get_dataloaders(ss.data_dir, batch_size, augment)
                trainer = Trainer(cfg, tr_loader, val_loader, checkpoint_dir=ss.ckpt_dir)
                if resume_path != "(train from scratch)":
                    trainer.load_checkpoint(resume_path)
                trainer.start()
                ss.trainer = trainer
                st.success("Training started.")
            except Exception as e:
                st.error(f"Could not start: {e}")

        if pause_clicked and ss.trainer:
            snap = ss.trainer.snapshot()
            (ss.trainer.resume if snap["status"] == "paused" else ss.trainer.pause)()

        if stop_clicked and ss.trainer:
            ss.trainer.stop()

        # ---------------- live status / charts ----------------
        if ss.trainer:
            snap = ss.trainer.snapshot()
            st.markdown(f"**Status:** `{snap['status']}`  ·  device: `{ss.trainer.device}`  "
                        f"·  best_val_acc: `{snap['best_val_acc']:.4f}` (epoch {snap['best_epoch']})")
            ep_progress = snap["epoch"] / max(snap["total_epochs"], 1)
            batch_progress = snap["batch"] / max(snap["total_batches"], 1)
            st.progress(min(1.0, ep_progress + batch_progress / max(snap["total_epochs"], 1)),
                        text=f"epoch {snap['epoch']+1}/{snap['total_epochs']} · batch {snap['batch']}/{snap['total_batches']}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("train loss", f"{snap['train_loss']:.4f}")
            m2.metric("train acc",  f"{snap['train_acc']:.4f}")
            m3.metric("val loss",   f"{snap['val_loss']:.4f}")
            m4.metric("val acc",    f"{snap['val_acc']:.4f}")

            if snap["history"]:
                hist = pd.DataFrame(snap["history"]).set_index("epoch")
                cc1, cc2 = st.columns(2)
                cc1.line_chart(hist[["train_loss", "val_loss"]])
                cc2.line_chart(hist[["train_acc", "val_acc"]])

            st.markdown("**Training log**")
            st.code("\n".join(snap["log_lines"][-30:]) or "(no log yet)", language="text")

            if snap["last_checkpoint"]:
                st.caption(f":floppy_disk: last: `{snap['last_checkpoint']}`")
            if snap["best_checkpoint"]:
                st.caption(f":star: best: `{snap['best_checkpoint']}`")

            if ss.trainer.is_alive():
                time.sleep(1.5)
                st.rerun()
        else:
            st.info("Configure hyperparameters in the sidebar, then press **Start**.")


# =========================================================== Evaluate tab

with tabs[1]:
    st.subheader("Evaluate a trained checkpoint on the test set")
    ckpts = list_checkpoints(ss.ckpt_dir)
    if not ckpts:
        st.info("No checkpoints found. Train a model first.")
    else:
        ckpt = st.selectbox("Checkpoint", ckpts, key="eval_ckpt")
        normalize = st.checkbox("Normalize confusion matrix rows", True)
        if st.button(":bar_chart: Run evaluation"):
            try:
                _, _, test_loader = get_dataloaders(ss.data_dir, 256, augment=False)
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                cfg = payload.get("config", {})
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = build_model(payload.get("model_name", cfg.get("model_name", "DeeperCNN")),
                                    activation=cfg.get("activation", "relu"),
                                    dropout=cfg.get("dropout", 0.3)).to(device)
                model.load_state_dict(payload["model_state"])

                with st.spinner("Running test set…"):
                    y_true, y_pred, y_proba = collect_predictions(model, test_loader, device)
                    metrics = compute_metrics(y_true, y_pred, y_proba)

                m1, m2 = st.columns(2)
                m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                m2.metric("Macro AUC", f"{metrics['macro_auc']:.4f}")

                report_df = pd.DataFrame(metrics["report"]).T
                st.dataframe(report_df.style.format(precision=3))

                cc1, cc2 = st.columns(2)
                cc1.pyplot(plot_confusion_matrix(y_true, y_pred, normalize=normalize))
                cc2.pyplot(plot_roc_curves(y_true, y_proba))
            except Exception as e:
                st.error(f"{e}")


# =========================================================== Predict tab

with tabs[2]:
    st.subheader("Recognise an emotion from an uploaded image")
    ckpts = list_checkpoints(ss.ckpt_dir)
    if not ckpts:
        st.info("No checkpoints found. Train a model first.")
    else:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            ckpt = st.selectbox("Checkpoint", ckpts, key="pred_ckpt")
            face_crop = st.checkbox("Detect & crop face", True)
            stylized = st.checkbox("Stylized / NPR mode", False,
                                    help="Boosts contrast & smooths line-art. Use for cel-shaded "
                                         "or anime-style character renders.")
            uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp"])
        with cc2:
            if uploaded is not None:
                img = Image.open(uploaded)
                st.image(img, caption="input", width=240)
                if st.button(":mag: Recognise"):
                    try:
                        res = predict_image(ckpt, img, face_crop=face_crop, stylized=stylized)
                        st.success(f"**{res['label']}**  (confidence {res['confidence']*100:.1f}%)")
                        probs_df = pd.Series(res["probabilities"]).sort_values(ascending=False)
                        st.bar_chart(probs_df)
                    except Exception as e:
                        st.error(f"{e}")


# =========================================================== About tab

with tabs[3]:
    st.markdown(
        """
### About this project

This implements Section 10.3 (Facial Emotion Recognition) of the homework.

| Spec requirement | Where it is implemented |
|---|---|
| Streamlit UI with upload / train / pause / resume / stop / log | `app.py` |
| At least 3 CNN architectures | `src/models.py` (`SimpleCNN`, `DeeperCNN`, `MiniResNet`) |
| Multi-framework support | PyTorch (CNNs) + Scikit-Learn (logreg baseline) |
| Hyperparameter selection (LR, optimizer, activation…) | sidebar widgets |
| Train-from-scratch and Resume | "Resume from" dropdown on Train tab |
| Checkpointing: best, last, on-pause, on-stop | `src/trainer.py` |
| Accuracy / confusion matrix / ROC | Evaluate tab |
| FER2013 preprocessing (grayscale, normalise, augment) | `src/dataset.py` |

**Stylized / NPR inference**: the Predict tab has a "Stylized / NPR mode" toggle.
FER2013 is photographic; cel-shaded characters look very different. The toggle
applies histogram equalisation + a small Gaussian blur before the standard
FER2013 transforms — usually enough to get a sensible reading on toon-shaded
character art, though best results require fine-tuning on a stylised dataset.
"""
    )