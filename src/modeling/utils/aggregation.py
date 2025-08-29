from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Tuple

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def _gem_pool(X: np.ndarray, p: float = 3.0, eps: float = 1e-12) -> np.ndarray:
    # GeM: ((1/M) * sum(x^p))^(1/p)
    Xp = np.power(np.clip(X, eps, None), p)
    pooled = np.power(np.mean(Xp, axis=0, keepdims=False), 1.0 / p)
    return pooled

_REDUCERS: Dict[str, Callable[..., np.ndarray]] = {
    "mean":  lambda X, **_: np.mean(X, axis=0),
    "median": lambda X, **_: np.median(X, axis=0),
    "max":   lambda X, **_: np.max(X, axis=0),
    "gem":   lambda X, **kw: _gem_pool(X, **kw),
}

class Aggregator:
    """
    Агрегация эмбеддингов:
    - по меткам (class prototypes)
    - редукторы: mean | median | max | gem(p)
    - опциональная L2-нормализация выхода
    """
    def __init__(self, reducer: str = "mean", p: float = 3.0, l2: bool = True) -> None:
        if reducer not in _REDUCERS:
            raise ValueError(f"Unknown reducer: {reducer}")
        self.reducer = reducer
        self.p = p
        self.l2 = l2

    def by_ids_with_labels(
        self,
        emb: np.ndarray,
        ids: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Агрегация эмбеддингов по уникальным идентификаторам моделей.
        На выходе: aggregated_emb, unique_ids, unique_labels
        """
        emb = np.asarray(emb)
        uniq = np.unique(ids)
        protos = []
        proto_ids = []
        proto_labels = []

        for u in uniq:
            idx = np.where(ids == u)[0]
            block = emb[idx]
            vec = _REDUCERS[self.reducer](block, p=self.p)
            protos.append(vec)
            proto_ids.append(u)
            # Предполагаем, что все изображения одной модели имеют одинаковый label
            proto_labels.append(labels[idx[0]])

        protos = np.vstack(protos)
        if self.l2:
            protos = l2_normalize(protos)
        return protos, proto_ids, proto_labels

    