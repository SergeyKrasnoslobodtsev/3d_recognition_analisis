import os, json, pickle
from dataclasses import dataclass, asdict
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from ..data.splits import stratified_pick_k_per_class
from ..data.normalization import scale_split
from ..metrics import classification_metrics_from_logits
from ..models.mlp_pooled import PooledMLP

@dataclass
class TrainConfigPooled:
    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.10
    early_stop_patience: int = 10
    grad_clip: float = 1.0
    ckpt_path: str = "best_pooled_mlp.pt"
    history_path: str = "history_pooled.json"
    val_per_class: int = 3
    # нормализация
    scaler: str = "standard"     # "standard" | "robust" | "power" | "none"
    robust_q: tuple = (25, 75)
    l2_norm: bool = True
    # модель
    width: int = 256
    depth: int = 2
    drop: float = 0.20

def train_pooled(
    X: np.ndarray,                 # (N, D)
    y: np.ndarray,                 # (N,)
    cfg: TrainConfigPooled,
    logger=None
):
    lg = logger if logger is not None else _noop_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lg.info(f"device={device}")

    le = LabelEncoder(); y_int = le.fit_transform(y)
    train_idx, val_idx = stratified_pick_k_per_class(y_int, k=cfg.val_per_class, seed=cfg.seed)
    X_tr, X_va, scaler = scale_split(X, train_idx, val_idx, scaler_type=cfg.scaler, robust_q=cfg.robust_q, l2_norm=cfg.l2_norm)
    lg.info(f"scaler={cfg.scaler}, l2_norm={cfg.l2_norm}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_int[train_idx], dtype=torch.long)),
                              batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_int[val_idx], dtype=torch.long)),
                              batch_size=cfg.batch_size, shuffle=False)

    in_dim = X.shape[1]; num_classes = len(le.classes_)
    model = PooledMLP(in_dim=in_dim, num_classes=num_classes, width=cfg.width, depth=cfg.depth, drop=cfg.drop).to(device)

    cls_counts = np.bincount(y_int[train_idx], minlength=num_classes)
    cls_w = (1.0 / np.clip(cls_counts, 1, None)); cls_w *= len(cls_w) / cls_w.sum()
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=cls_w_t, label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss, best_state, patience = float("inf"), None, 0

    for epoch in tqdm(range(cfg.num_epochs), desc="epochs"):
        model.train()
        sum_loss, correct, total = 0.0, 0, 0
        for xb, yb in tqdm(train_loader, desc=f"train {epoch+1}/{cfg.num_epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(xb); loss = criterion(logits, yb)
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler_amp.step(optimizer); scaler_amp.update()
            sum_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        tr_loss = sum_loss / max(1, total); tr_acc = correct / max(1, total)

        model.eval()
        sum_loss, correct, total = 0.0, 0, 0
        logits_all, y_all = [], []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="valid", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(xb); loss = criterion(logits, yb)
                sum_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
                logits_all.append(logits.detach().cpu().numpy()); y_all.append(yb.detach().cpu().numpy())
        va_loss = sum_loss / max(1, total); va_acc = correct / max(1, total)
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc);  history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        lg.info(f"[{epoch+1:03d}] loss {tr_loss:.4f}/{va_loss:.4f} acc {tr_acc:.4f}/{va_acc:.4f}")

        if va_loss < best_val_loss - 1e-6:
            best_val_loss, patience = va_loss, 0
            best_state = {"model": model.state_dict(), "epoch": epoch, "val_loss": best_val_loss, "config": asdict(cfg)}
            torch.save(best_state, cfg.ckpt_path); lg.info(f"ckpt saved → {cfg.ckpt_path}")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience: lg.warning("early stopping"); break

    with open(cfg.history_path, "w", encoding="utf-8") as f: json.dump(history, f, ensure_ascii=False, indent=2)
    if best_state is not None: model.load_state_dict(best_state["model"])

    # финальные метрики
    logits_all = np.concatenate(logits_all, 0); y_all = np.concatenate(y_all, 0)
    metrics = classification_metrics_from_logits(logits_all, y_all)
    lg.info(f"final metrics: {metrics}")

    # артефакты инференса
    with open(os.path.splitext(cfg.ckpt_path)[0] + "_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.splitext(cfg.ckpt_path)[0] + "_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return model, le, history, metrics

def _noop_logger():
    class _L:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
    return _L()