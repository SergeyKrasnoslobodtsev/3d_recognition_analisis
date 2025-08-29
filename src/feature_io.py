import numpy as np
from pathlib import Path
from typing import Tuple

def save_features(features: np.ndarray, labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        labels=labels
    )

def load_features(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    features = data["features"]
    labels = data["labels"]
    return features, labels