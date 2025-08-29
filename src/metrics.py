import numpy as np
from sklearn.metrics import silhouette_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

def mean_average_precision(X: np.ndarray, y: np.ndarray) -> float:
    """MAP для многоклассовой задачи"""
    unique_labels = np.unique(y)
    encoded_labels = LabelEncoder().fit_transform(y)
    if len(unique_labels) < 2:
        return 0
    try:
        y_binary = label_binarize(y, classes=unique_labels)
        min_class_count = np.min(np.bincount(encoded_labels))
        n_splits = min(3, min_class_count)
        if n_splits < 2:
            return 0
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
        y_scores = np.zeros_like(y_binary, dtype=float)
        for train_index, test_index in cv.split(X, encoded_labels):
            X_train, X_test = X[train_index], X[test_index]
            y_train_binary = y_binary[train_index]
            classifier.fit(X_train, y_train_binary)
            y_scores[test_index] = classifier.predict_proba(X_test)
        ap_scores = []
        for i in range(len(unique_labels)):
            if len(np.unique(y_binary[:, i])) > 1:
                ap = average_precision_score(y_binary[:, i], y_scores[:, i])
                ap_scores.append(ap)
        return np.mean(ap_scores) if ap_scores else 0
    except Exception:
        return 0

def silhouette(X: np.ndarray, y: np.ndarray, sample_size=10000) -> float:
    """Silhouette Score"""
    unique_labels = np.unique(y)
    encoded_labels = LabelEncoder().fit_transform(y)
    if len(unique_labels) < 2:
        return 0
    features_to_eval = X
    labels_to_eval = encoded_labels
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        features_to_eval = X[idx]
        labels_to_eval = encoded_labels[idx]
    try:
        return silhouette_score(features_to_eval, labels_to_eval)
    except Exception:
        return 0

def intra_inter_distances(X: np.ndarray, y: np.ndarray) -> dict:
    """Внутри- и межклассовые расстояния"""
    unique_labels = np.unique(y)
    intra_distances = []
    centroids = {}
    for label in unique_labels:
        mask = y == label
        centroids[label] = np.mean(X[mask], axis=0)
    for label in unique_labels:
        mask = y == label
        class_features = X[mask]
        centroid = centroids[label]
        distances = np.linalg.norm(class_features - centroid, axis=1)
        intra_distances.extend(distances)
    centroid_list = list(centroids.values())
    inter_distances = []
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            distance = np.linalg.norm(centroid_list[i] - centroid_list[j])
            inter_distances.append(distance)
    return {
        "intra_class": np.mean(intra_distances) if intra_distances else 0,
        "inter_class": np.mean(inter_distances) if inter_distances else 0
    }

def separation_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Соотношение межклассового и внутриклассового расстояния"""
    dists = intra_inter_distances(X, y)
    intra = dists["intra_class"]
    inter = dists["inter_class"]
    return inter / (intra + 1e-9) if intra > 0 else 0

def get_metrics(X: np.ndarray, y: np.ndarray) -> dict:
    """Собирает все метрики в словарь"""
    dists = intra_inter_distances(X, y)
    return {
        "map": mean_average_precision(X, y),
        "silhouette": silhouette(X, y),
        "intra_class_distance": dists["intra_class"],
        "inter_class_distance": dists["inter_class"],
        "separation_ratio": separation_ratio(X, y),
        "samples": len(X),
        "classes": len(np.unique(y)),
        "feature_dim": X.shape[1] if len(X) > 0 else 0
    }