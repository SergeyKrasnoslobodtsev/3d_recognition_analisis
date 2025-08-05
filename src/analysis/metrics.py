import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine, euclidean
from typing import List, Dict, Tuple
from collections import defaultdict

from loguru import logger
from ..features.extractor import FeatureDataset

class FeatureAnalysisMetrics:
    """Метрики для анализа качества признаков"""
    
    @staticmethod
    def extract_features_and_labels(feature_dataset: FeatureDataset) -> Tuple[np.ndarray, List[str]]:
        """Извлекает массив признаков и меток классов из FeatureDataset"""
        features = np.array([fv.feature_vector for fv in feature_dataset.features])
        labels = [fv.detail_type for fv in feature_dataset.features]
        return features, labels
    
    @staticmethod
    def compute_intra_class_similarity(features: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """Вычисляет внутриклассовую схожесть (средняя cosine similarity внутри класса)"""
        class_similarities = {}
        
        # Группируем признаки по классам
        class_features = defaultdict(list)
        for feature, label in zip(features, labels):
            class_features[label].append(feature)
        
        for class_name, class_feature_list in class_features.items():
            if len(class_feature_list) < 2:
                class_similarities[class_name] = 1.0  # Один элемент - максимальная схожесть
                continue
                
            similarities = []
            class_features_array = np.array(class_feature_list)
            
            # Вычисляем все попарные cosine similarities
            for i in range(len(class_features_array)):
                for j in range(i + 1, len(class_features_array)):
                    similarity = 1 - cosine(class_features_array[i], class_features_array[j])
                    similarities.append(similarity)
            
            class_similarities[class_name] = np.mean(similarities)
        
        return class_similarities
    
    @staticmethod
    def compute_inter_class_distance(features: np.ndarray, labels: List[str]) -> Dict[Tuple[str, str], float]:
        """Вычисляет межклассовые расстояния (средние cosine distances между классами)"""
        # Группируем признаки по классам
        class_features = defaultdict(list)
        for feature, label in zip(features, labels):
            class_features[label].append(feature)
        
        # Вычисляем центроиды классов
        class_centroids = {}
        for class_name, class_feature_list in class_features.items():
            class_centroids[class_name] = np.mean(class_feature_list, axis=0)
        
        # Вычисляем расстояния между центроидами
        inter_class_distances = {}
        class_names = list(class_centroids.keys())
        
        for i in range(len(class_names)):
            for j in range(i + 1, len(class_names)):
                class1, class2 = class_names[i], class_names[j]
                distance = cosine(class_centroids[class1], class_centroids[class2])
                inter_class_distances[(class1, class2)] = distance
        
        return inter_class_distances
    
    @staticmethod
    def clustering_quality_metrics(features: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """Метрики качества кластеризации"""
        n_clusters = len(set(labels))
        
        if n_clusters < 2:
            logger.warning("Недостаточно классов для вычисления метрик кластеризации")
            return {}
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        metrics = {
            'silhouette_score': silhouette_score(features, cluster_labels),
            'calinski_harabasz_score': calinski_harabasz_score(features, cluster_labels),
            'inertia': kmeans.inertia_,
            'n_clusters': n_clusters
        }
        
        return metrics
    
    @staticmethod
    def compute_class_separability(features: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """Вычисляет разделимость классов"""
        intra_class_sim = FeatureAnalysisMetrics.compute_intra_class_similarity(features, labels)
        inter_class_dist = FeatureAnalysisMetrics.compute_inter_class_distance(features, labels)
        
        # Средняя внутриклассовая схожесть
        avg_intra_similarity = np.mean(list(intra_class_sim.values()))
        
        # Средняя межклассовая дистанция
        avg_inter_distance = np.mean(list(inter_class_dist.values()))
        
        # Индекс разделимости (выше = лучше)
        separability_index = avg_inter_distance / (1 - avg_intra_similarity + 1e-6)
        
        return {
            'avg_intra_class_similarity': avg_intra_similarity,
            'avg_inter_class_distance': avg_inter_distance,
            'separability_index': separability_index
        }
    
    @staticmethod
    def analyze_feature_dataset(feature_dataset: FeatureDataset) -> Dict[str, any]:
        """Полный анализ датасета признаков"""
        logger.info(f"Анализ признаков для экстрактора: {feature_dataset.extractor_type}")
        
        features, labels = FeatureAnalysisMetrics.extract_features_and_labels(feature_dataset)
        
        results = {
            'extractor_type': feature_dataset.extractor_type,
            'feature_dimension': feature_dataset.feature_dimension,
            'n_samples': len(features),
            'n_classes': len(set(labels)),
            'class_distribution': {label: labels.count(label) for label in set(labels)}
        }
        
        # Добавляем метрики
        results.update(FeatureAnalysisMetrics.clustering_quality_metrics(features, labels))
        results.update(FeatureAnalysisMetrics.compute_class_separability(features, labels))
        results['intra_class_similarities'] = FeatureAnalysisMetrics.compute_intra_class_similarity(features, labels)
        results['inter_class_distances'] = FeatureAnalysisMetrics.compute_inter_class_distance(features, labels)
        
        return results