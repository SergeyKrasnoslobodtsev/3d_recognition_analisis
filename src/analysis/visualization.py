import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from loguru import logger
from ..features.extractor import FeatureDataset
from .metrics import FeatureAnalysisMetrics

# Настройка стилей
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FeatureVisualizer:
    """Визуализация признаков и результатов анализа"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Визуализации будут сохранены в: {self.save_dir}")
    
    def plot_feature_distribution(self, feature_dataset: FeatureDataset, save: bool = True) -> None:
        """Распределение признаков по классам"""
        features, labels = FeatureAnalysisMetrics.extract_features_and_labels(feature_dataset)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Feature Distribution Analysis - {feature_dataset.extractor_type.upper()}', fontsize=16)
        
        # 1. Среднее значение признаков по классам
        class_means = {}
        unique_labels = list(set(labels))
        
        for label in unique_labels:
            mask = [l == label for l in labels]
            class_features = features[mask]
            class_means[label] = np.mean(class_features, axis=0)
        
        # Boxplot средних значений
        means_data = list(class_means.values())
        axes[0, 0].boxplot(means_data, labels=list(class_means.keys()))
        axes[0, 0].set_title('Feature Means by Class')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Стандартное отклонение по классам
        class_stds = {}
        for label in unique_labels:
            mask = [l == label for l in labels]
            class_features = features[mask]
            class_stds[label] = np.std(class_features, axis=0)
        
        stds_data = list(class_stds.values())
        axes[0, 1].boxplot(stds_data, labels=list(class_stds.keys()))
        axes[0, 1].set_title('Feature Standard Deviations by Class')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Распределение классов
        class_counts = pd.Series(labels).value_counts()
        axes[1, 0].bar(class_counts.index, class_counts.values)
        axes[1, 0].set_title('Class Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Корреляция признаков (первые 50 для читаемости)
        if features.shape[1] > 50:
            corr_features = features[:, :50]
        else:
            corr_features = features
            
        correlation_matrix = np.corrcoef(corr_features.T)
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Feature Correlation Matrix (first 50 dims)')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{feature_dataset.extractor_type}_feature_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.show()
    
    def plot_dimensionality_reduction(self, feature_dataset: FeatureDataset, 
                                    methods: List[str] = ['pca', 'tsne'], 
                                    save: bool = True) -> None:
        """Визуализация с понижением размерности"""
        features, labels = FeatureAnalysisMetrics.extract_features_and_labels(feature_dataset)
        
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle(f'Dimensionality Reduction - {feature_dataset.extractor_type.upper()}', fontsize=16)
        
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for idx, method in enumerate(methods):
            if method.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                reduced_features = reducer.fit_transform(features)
                title = f'PCA (explained var: {reducer.explained_variance_ratio_.sum():.2f})'
                
            elif method.lower() == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
                reduced_features = reducer.fit_transform(features)
                title = 't-SNE'
            
            # Plotting
            for i, label in enumerate(unique_labels):
                mask = [l == label for l in labels]
                axes[idx].scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                                c=[colors[i]], label=label, alpha=0.7, s=50)
            
            axes[idx].set_title(title)
            axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{feature_dataset.extractor_type}_dimensionality_reduction.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.show()
    
    def plot_similarity_heatmap(self, feature_dataset: FeatureDataset, save: bool = True) -> None:
        """Тепловая карта схожести между классами"""
        features, labels = FeatureAnalysisMetrics.extract_features_and_labels(feature_dataset)
        
        # Вычисляем центроиды классов
        unique_labels = sorted(list(set(labels)))
        centroids = []
        
        for label in unique_labels:
            mask = [l == label for l in labels]
            class_features = features[mask]
            centroids.append(np.mean(class_features, axis=0))
        
        centroids = np.array(centroids)
        
        # Вычисляем матрицу схожести (cosine similarity)
        similarity_matrix = np.zeros((len(unique_labels), len(unique_labels)))
        
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = 1 - np.dot(centroids[i], centroids[j]) / (
                        np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j])
                    )
                    similarity_matrix[i, j] = 1 - similarity  # Преобразуем distance в similarity
        
        # Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=unique_labels, 
                   yticklabels=unique_labels,
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title(f'Class Similarity Matrix - {feature_dataset.extractor_type.upper()}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{feature_dataset.extractor_type}_similarity_heatmap.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.show()