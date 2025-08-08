import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap
from loguru import logger

from .config import FIGURES_DIR


def plot_similarity_heatmap(features: list[np.ndarray], labels: list[str], name_file: str | None) -> None:
    """Тепловая карта схожести между классами"""

    unique_labels = sorted(list(set(labels)))
    centroids = []

    features_array = np.array(features)
    labels_array = np.array(labels)
    
    for label in unique_labels:
        # Находим все индексы для данного класса
        mask = labels_array == label
        class_features = features_array[mask]
        # Вычисляем центроид как среднее по всем образцам класса
        centroid = np.mean(class_features, axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    similarity_matrix = cosine_similarity(centroids)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
               xticklabels=unique_labels, 
               yticklabels=unique_labels,
               annot=True, 
               fmt='.3f',
               cmap='crest')
    
    plt.title('Матрица схожести классов (по центроидам)')
    
    _save_plot(plt.gcf(), 'heatmap', name_file)
    plt.show()


def plot_matrix_error(features: list[np.ndarray], labels: list[str], name_file: str | None) -> None:
    """
    Матрица ошибок для классификации.
    Использует KNN для классификации и строит матрицу ошибок.
    """

    features_array = np.array(features)
    labels_array = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, 
        labels_array, 
        test_size=0.3, 
        random_state=42,     
        stratify=labels_array 
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    

    cm = confusion_matrix(y_test, y_pred)
    unique_labels = sorted(list(set(labels)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
               annot=True, 
               fmt='d', 
               cmap='Blues',
               xticklabels=unique_labels,
               yticklabels=unique_labels)
    
    plt.title('Матрица ошибок (KNN классификация)')
    plt.ylabel('Истинные классы')
    plt.xlabel('Предсказанные классы')
    
    _save_plot(plt.gcf(), 'confusion_matrix', name_file)
    plt.show()


def plot_clustering_tsne(features: list[np.ndarray], labels: list[str], name_file: str | None) -> None:
    """Визуализация t-SNE"""
    features_array = np.array(features)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_array)
    
    plt.figure(figsize=(12, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7)
    
    plt.title('t-SNE визуализация признаков')
    plt.legend()
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    _save_plot(plt.gcf(), 'tsne', name_file)
    plt.show()


def plot_clustering_umap(features: list[np.ndarray], labels: list[str], name_file: str | None) -> None:
    """Визуализация UMAP"""
    features_array = np.array(features)
    

    reducer = umap.UMAP(random_state=42)
    features_2d = reducer.fit_transform(features_array)
    
    plt.figure(figsize=(12, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7)
    
    plt.title('UMAP визуализация признаков')
    plt.legend()
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    _save_plot(plt.gcf(), 'umap', name_file)
    plt.show()


def _save_plot(fig: plt.Figure, prefix: str, name_file: str | None) -> None:
    """Сохраняет график в файл"""
    if name_file:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        filepath = FIGURES_DIR / f"{name_file}_{prefix}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"График сохранен: {filepath}")