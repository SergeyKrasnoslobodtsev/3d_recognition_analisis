import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, normalize as l2_normalize

# pooled (N,D)
def scale_split(X: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                scaler_type: str = "standard", robust_q=(25,75), l2_norm=True):
    X_train = X[train_idx].astype(np.float32, copy=False)
    X_val   = X[val_idx].astype(np.float32, copy=False)
    scaler = None
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler(quantile_range=robust_q)
    elif scaler_type == "power":
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
    elif scaler_type == "none":
        pass
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")
    if scaler is not None:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
    if l2_norm:
        X_train = l2_normalize(X_train, norm="l2")
        X_val   = l2_normalize(X_val, norm="l2")
    return X_train, X_val, scaler

# matrix (N,7,L)
def fit_norm_stats_cp(X_train: np.ndarray):
    mu  = X_train.mean(axis=0, keepdims=True).astype(np.float32)  # (1,7,L)
    std = (X_train.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    return mu, std

def apply_norm_cp(X: np.ndarray, mu: np.ndarray, std: np.ndarray):
    return ((X - mu) / std).astype(np.float32)