import numpy as np

from sklearn.calibration import label_binarize
from sklearn.metrics import (average_precision_score, silhouette_score)
from sklearn.model_selection import cross_val_predict
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier


class Metrics:

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)

    def get_stats(self) -> dict:
        distances = self.intra_inter_distances()
        intra_dist = distances["intra_class"]
        inter_dist = distances["inter_class"]
        separation_ratio = inter_dist / (intra_dist + 1e-9) if intra_dist > 0 else 0
        return {
            "map_at_10": self.mean_average_precision(),
            "intra_class": distances["intra_class"],
            "inter_class": distances["inter_class"],
            "separation_ratio": separation_ratio,
            "clustering_scores": self.silhouette_score()
        }

    def mean_average_precision(self):
        """MAP для многоклассовой задачи"""
        if len(self.unique_labels) < 2:
            return 0
        try:
            y_binary = label_binarize(self.labels, classes=self.unique_labels)
            min_class_count = np.min(np.bincount(self.labels))
            n_splits = min(3, min_class_count)
            
            if n_splits < 2:
                return 0

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
            

            y_scores = cross_val_predict(classifier, self.features, self.labels, cv=cv, method='predict_proba')

            for train_index, test_index in cv.split(self.features, self.labels):
                X_train, X_test = self.features[train_index], self.features[test_index]
                # Для обучения используется y_binary
                y_train_binary = y_binary[train_index]
                
                classifier.fit(X_train, y_train_binary)
                y_scores[test_index] = classifier.predict_proba(X_test)

            ap_scores = []
            for i in range(len(self.unique_labels)):
                if len(np.unique(y_binary[:, i])) > 1:
                    ap = average_precision_score(y_binary[:, i], y_scores[:, i])
                    ap_scores.append(ap)
            
            return np.mean(ap_scores) if ap_scores else 0
        except ValueError as e:
            return 0

    def silhouette_score(self, sample_size=10000):
        """Silhouette Score. Для ускорения используется выборка."""
        if len(self.unique_labels) < 2:
            return 0
        
        features_to_eval = self.features
        labels_to_eval = self.labels

        if len(self.features) > sample_size:
            idx = np.random.choice(len(self.features), sample_size, replace=False)
            features_to_eval = self.features[idx]
            labels_to_eval = self.labels[idx]

        try:
            return silhouette_score(features_to_eval, labels_to_eval)
        except ValueError as e:
            return 0

    def intra_inter_distances(self):
        """Внутри- и межклассовые расстояния"""
        intra_distances = []
        centroids = {}
        
        # Сначала вычисляем центроиды для всех классов
        for label in self.unique_labels:
            mask = self.labels == label
            centroids[label] = np.mean(self.features[mask], axis=0)

        # Внутриклассовые расстояния (от точек до центроида)
        for label in self.unique_labels:
            mask = self.labels == label
            class_features = self.features[mask]
            centroid = centroids[label]
            # Расстояние от каждой точки класса до центроида этого класса
            distances = np.linalg.norm(class_features - centroid, axis=1)
            intra_distances.extend(distances)
        
        # Межклассовые расстояния (между центроидами)
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
    
    def plot_umap(self, metric: str = "cosine"):
        """
        Улучшенная UMAP-визуализация эмбеддингов в 2D.
        """
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        reducer = umap.UMAP(
            metric=metric,
            random_state=42,
            n_jobs=1,
        )
        xy = reducer.fit_transform(self.features)

        # Цвета для классов
        lut = {u: i for i, u in enumerate(self.unique_labels)}
        c = np.array([lut[l] for l in self.labels])

        plt.figure(figsize=(16, 16))
        plt.scatter(
            xy[:, 0], xy[:, 1],
            c=c,
            cmap="tab20",
            s=120,            # увеличенный размер точек
            alpha=0.85,
            edgecolors='w',   # белая окантовка
            linewidths=0.7
        )
        plt.title("UMAP: 2D визуализация классов", fontsize=22)
        plt.xlabel("UMAP-1", fontsize=16)
        plt.ylabel("UMAP-2", fontsize=16)
        plt.grid(alpha=0.2)

        # Легенда по классам (максимум 20 для читаемости)
        handles = [
            mpatches.Patch(color=plt.cm.tab20(i / max(1, len(self.unique_labels)-1)), label=str(u))
            for i, u in enumerate(self.unique_labels[:20])
        ]
        plt.legend(handles=handles, title="Классы (первые 20)", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

        plt.tight_layout()
        plt.show()

