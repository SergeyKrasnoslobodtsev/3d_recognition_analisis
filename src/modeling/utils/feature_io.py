import numpy as np
import torch
from pathlib import Path
from typing import Tuple

def save_features(features: np.ndarray, labels: np.ndarray, ids: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        ids=ids
    )

def load_features(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    features = data["features"]
    labels = data["labels"]
    ids = data["ids"] # по этим данным будем агрегировать
    return features, labels, ids