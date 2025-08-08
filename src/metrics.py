from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, average_precision_score
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, average_precision_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier


class FeatureMetrics:
    """Метрики для анализа признаков"""
    def __init__(self, features: list[np.ndarray], labels: list[str]):
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.unique_labels = np.unique(self.labels)
        self.encoded_labels = LabelEncoder().fit_transform(self.labels)

    def mean_average_precision(self):
        """MAP для многоклассовой задачи"""
        if len(self.unique_labels) < 2:
            return 0
        try:
            y_binary = label_binarize(self.labels, classes=self.unique_labels)
            min_class_count = np.min(np.bincount(self.encoded_labels))
            n_splits = min(3, min_class_count)
            
            if n_splits < 2:
                logger.warning("Невозможно выполнить cross-валидацию: слишком мало сэмплов в классе.")
                return 0

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
            
            # Ручная реализация cross-валидации для обработки разных форматов y
            y_scores = np.zeros_like(y_binary, dtype=float)

            # cv.split использует self.encoded_labels для стратификации
            for train_index, test_index in cv.split(self.features, self.encoded_labels):
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
            logger.error(f"Ошибка при расчете MAP: {e}")
            return 0

    def silhouette_score(self, sample_size=10000):
        """Silhouette Score. Для ускорения используется выборка."""
        if len(self.unique_labels) < 2:
            return 0
        
        features_to_eval = self.features
        labels_to_eval = self.encoded_labels

        if len(self.features) > sample_size:
            logger.debug(f"Для Silhouette используется выборка размером {sample_size}")
            idx = np.random.choice(len(self.features), sample_size, replace=False)
            features_to_eval = self.features[idx]
            labels_to_eval = self.encoded_labels[idx]

        try:
            return silhouette_score(features_to_eval, labels_to_eval)
        except ValueError as e:
            logger.error(f"Ошибка при расчете Silhouette: {e}")
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

    def get_metrics(self):
        """Возвращает все метрики в виде словаря"""
        distances = self.intra_inter_distances()
        intra_dist = distances["intra_class"]
        inter_dist = distances["inter_class"]
        
        # Рассчитываем соотношение для нормализованной оценки.
        # Добавляем небольшое число к знаменателю для избежания деления на ноль.
        separation_ratio = inter_dist / (intra_dist + 1e-9) if intra_dist > 0 else 0

        return {
            "map": self.mean_average_precision(),
            "silhouette": self.silhouette_score(),
            "intra_class_distance": intra_dist,
            "inter_class_distance": inter_dist,
            "separation_ratio": separation_ratio,
            "samples": len(self.features),
            "classes": len(self.unique_labels),
            "feature_dim": self.features.shape[1] if len(self.features) > 0 else 0
        }

    def print_all_metrics(self):
        """Простой вывод метрик"""
        metrics = self.get_metrics()
        logger.info(f"MAP: {metrics['map']:.3f}")
        logger.info(f"Silhouette: {metrics['silhouette']:.3f}")
        logger.info(f"Separation Ratio (Inter/Intra): {metrics['separation_ratio']:.3f}")
        logger.info(f"  - Intra-class distance: {metrics['intra_class_distance']:.3f}")
        logger.info(f"  - Inter-class distance: {metrics['inter_class_distance']:.3f}")