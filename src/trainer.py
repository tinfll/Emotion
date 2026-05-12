"""Background-thread Trainer with pause/resume/stop, checkpointing, and live metrics.

Designed for Streamlit: the UI thread reads `trainer.state` snapshots while a
worker thread does the actual training.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import build_model, count_parameters


OPTIMIZERS = {
    "Adam":    lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
    "AdamW":   lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
    "SGD":     lambda p, lr, wd: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True),
    "RMSprop": lambda p, lr, wd: torch.optim.RMSprop(p, lr=lr, weight_decay=wd),
}


@dataclass
class TrainConfig:
    model_name: str = "DeeperCNN"
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    activation: str = "relu"
    dropout: float = 0.4
    batch_size: int = 64
    epochs: int = 15
    augment: bool = True
    label_smoothing: float = 0.05
    scheduler: str = "cosine"


@dataclass
class TrainState:
    status: str = "idle"            # idle | running | paused | stopping | finished | error
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    total_batches: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    best_val_acc: float = 0.0
    best_epoch: int = -1
    history: list[dict] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    error: str = ""
    last_checkpoint: str = ""
    best_checkpoint: str = ""


class Trainer:
    def __init__(self, config: TrainConfig, train_loader: DataLoader, val_loader: DataLoader,
                 checkpoint_dir: str = "checkpoints", device: Optional[str] = None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.state = TrainState(total_epochs=config.epochs, total_batches=len(train_loader))
        self._lock = threading.Lock()
        self._pause_event = threading.Event(); self._pause_event.set()  # set => not paused
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.model = build_model(config.model_name, activation=config.activation, dropout=config.dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = OPTIMIZERS[config.optimizer](self.model.parameters(), config.learning_rate, config.weight_decay)
        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        elif config.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, config.epochs // 3), gamma=0.5)
        else:
            self.scheduler = None

        self._log(f"Built {config.model_name} ({count_parameters(self.model):,} params) on {self.device}")

    # ------------------------------- public control -------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear(); self._pause_event.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self):
        self._pause_event.clear()
        with self._lock:
            self.state.status = "paused"
        self._log("Pause requested — will pause at next batch boundary.")

    def resume(self):
        self._pause_event.set()
        with self._lock:
            self.state.status = "running"
        self._log("Resumed.")

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()  # unblock pause if any
        with self._lock:
            self.state.status = "stopping"
        self._log("Stop requested — will halt at next batch boundary.")

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> dict:
        with self._lock:
            return asdict(self.state)

    # ------------------------------- checkpointing -------------------------------

    def save_checkpoint(self, tag: str) -> str:
        path = self.checkpoint_dir / f"{self.config.model_name}_{tag}.pth"
        payload = {
            "model_name": self.config.model_name,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.state.epoch,
            "config": asdict(self.config),
            "best_val_acc": self.state.best_val_acc,
            "history": self.state.history,
        }
        torch.save(payload, path)
        return str(path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        with self._lock:
            self.state.epoch = ckpt.get("epoch", 0)
            self.state.history = ckpt.get("history", [])
            self.state.best_val_acc = ckpt.get("best_val_acc", 0.0)
        self._log(f"Resumed from checkpoint: {path} (epoch {self.state.epoch}, best_val_acc={self.state.best_val_acc:.4f})")

    # ------------------------------- training loop -------------------------------

    def _log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            self.state.log_lines.append(line)
            if len(self.state.log_lines) > 500:
                self.state.log_lines = self.state.log_lines[-500:]

    def _run(self):
        try:
            with self._lock:
                self.state.status = "running"
                self.state.total_epochs = self.config.epochs
                self.state.total_batches = len(self.train_loader)
            start_epoch = self.state.epoch
            for epoch in range(start_epoch, self.config.epochs):
                if self._stop_event.is_set():
                    break
                with self._lock:
                    self.state.epoch = epoch
                    self.state.batch = 0
                self._train_one_epoch(epoch)
                if self._stop_event.is_set():
                    break
                val_loss, val_acc = self._evaluate()
                with self._lock:
                    self.state.val_loss = val_loss
                    self.state.val_acc = val_acc
                    self.state.history.append({
                        "epoch": epoch + 1,
                        "train_loss": self.state.train_loss,
                        "train_acc": self.state.train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    improved = val_acc > self.state.best_val_acc
                    if improved:
                        self.state.best_val_acc = val_acc
                        self.state.best_epoch = epoch + 1
                if improved:
                    self.state.best_checkpoint = self.save_checkpoint("best")
                    self._log(f"epoch {epoch+1}: new best val_acc={val_acc:.4f} → saved best.")
                self.state.last_checkpoint = self.save_checkpoint("last")
                if self.scheduler:
                    self.scheduler.step()
                self._log(f"epoch {epoch+1}/{self.config.epochs}  "
                          f"train_loss={self.state.train_loss:.4f} train_acc={self.state.train_acc:.4f}  "
                          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            with self._lock:
                if self._stop_event.is_set():
                    # Auto-save on stop, per the spec.
                    self.state.last_checkpoint = self.save_checkpoint("stopped")
                    self.state.status = "finished"
                    self._log("Training stopped. Checkpoint saved.")
                else:
                    self.state.status = "finished"
                    self._log("Training finished.")
        except Exception as e:
            with self._lock:
                self.state.status = "error"
                self.state.error = str(e)
            self._log(f"ERROR: {e}")

    def _wait_if_paused(self):
        # If pause requested, save once and block until resumed.
        if not self._pause_event.is_set():
            self.state.last_checkpoint = self.save_checkpoint("pause")
            self._log("Auto-saved checkpoint on pause.")
            self._pause_event.wait()

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for batch_idx, (x, y) in enumerate(self.train_loader):
            if self._stop_event.is_set():
                return
            self._wait_if_paused()
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)

            with self._lock:
                self.state.batch = batch_idx + 1
                self.state.train_loss = running_loss / seen
                self.state.train_acc = correct / seen

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in self.val_loader:
            x = x.to(self.device); y = y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)
        return loss_sum / max(seen, 1), correct / max(seen, 1)


def list_checkpoints(checkpoint_dir: str = "checkpoints") -> list[str]:
    p = Path(checkpoint_dir)
    if not p.exists():
        return []
    return sorted(str(c) for c in p.glob("*.pth"))
