import numpy as np
from typing import Tuple

def stratified_pick_k_per_class(y: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, y_idx = np.unique(y, return_inverse=True)
    val_idx = []
    for c in range(len(classes)):
        c_idx = np.where(y_idx == c)[0]
        rng.shuffle(c_idx)
        take = min(k, len(c_idx))
        val_idx.extend(c_idx[:take])
    val_idx = np.array(sorted(set(val_idx)))
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]
    return train_idx, val_idx