import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from typing import List, Tuple, Optional

class SimpleAttentionPool:
    """
    Взвешенное усреднение по видам:
      s_i = x_i @ q
      alpha = softmax(s)
      z = sum_i alpha_i * x_i
    """
    def __init__(self, dim: int, q: Optional[np.ndarray] = None):
        self.dim = dim
        self.q = np.random.randn(dim).astype(np.float32) if q is None else q.astype(np.float32)

    def pool(self, X: np.ndarray) -> np.ndarray:
        # X: (num_views, dim)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"X shape {X.shape} != (*, {self.dim})")
        scores = X @ self.q  # (num_views,)
        # стабильный softmax
        scores = scores - scores.max()
        w = np.exp(scores)
        w /= (w.sum() + 1e-9)
        return (w[:, None] * X).sum(axis=0)

    def save(self, path: str):
        np.save(path, self.q)

    @staticmethod
    def load(path: str) -> "SimpleAttentionPool":
        q = np.load(path)
        return SimpleAttentionPool(dim=q.shape[0], q=q)

class AttnPoolTrainer:
    """
    Мини-тренер: учим q и линейный классификатор на pooled эмбеддингах.
    Не трогаем CLIP/DINO — только q и C.
    """
    def __init__(self, dim: int, num_classes: int, lr: float = 1e-2, weight_decay: float = 1e-4, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.q = nn.Parameter(torch.randn(dim))
        self.cls = nn.Linear(dim, num_classes)
        self.opt = optim.Adam([self.q] + list(self.cls.parameters()), lr=lr, weight_decay=weight_decay)
        self.ce = nn.CrossEntropyLoss()

    def _pool(self, X: torch.Tensor) -> torch.Tensor:
        # X: (V, D)
        s = X @ self.q  # (V,)
        a = torch.softmax(s, dim=0)
        z = (a[:, None] * X).sum(dim=0)
        return z

    def fit(self, batches: List[Tuple[np.ndarray, int]], epochs: int = 10) -> np.ndarray:
        """
        batches: список (X_views, y), где X_views.shape == (num_views, dim)
        Возвращает обученный вектор q как np.ndarray.
        """
        self.q.data = self.q.data.to(self.device)
        self.cls.to(self.device)
        for ep in range(epochs):
            total = 0.0
            for Xv, y in batches:
                X = torch.from_numpy(Xv).float().to(self.device)
                y_t = torch.tensor(y, dtype=torch.long, device=self.device)
                z = self._pool(X)  # (D,)
                logits = self.cls(z[None, :])  # (1, C)
                loss = self.ce(logits, y_t[None])
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                total += loss.item()
            logger.info(f"[AttnPoolTrainer] epoch {ep+1}/{epochs} loss={total/len(batches):.4f}")
        return self.q.detach().cpu().numpy()