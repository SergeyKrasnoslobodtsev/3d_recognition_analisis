from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
import pickle

from loguru import logger
import torch


import numpy as np

from ..dataset import DataModel

class ExtractorType(str, Enum):
    CLIP_B32 = "clip_b32"
    CLIP_B16 = "clip_b16"
    CLIP_L14 = "clip_l14"
    DINO_BASE = "dino_base"
    GOOGLE_B16_384 = "google_b16_384"
    GOOGLE_B16_224 = "google_b16_224"

    BREP = "brep"

@dataclass
class FeatureVector:
    """Вектор признаков"""
    model_id: str
    vector: Any
    label: str

@dataclass
class FeatureDataset:
    """Датасет признаков"""
    features: list[FeatureVector]
    extractor_type: str

    def add_vector(self, model_id: str, feature_vector: Any, label: str) -> None:
        self.features.append(FeatureVector(model_id=model_id, vector=feature_vector, label=label))

    def __len__(self) -> int:
        return len(self.features)

class FeatureExtractor(ABC):
    """Абстрактный класс для извлечения признаков"""
    
    def __init__(self, name: str, model: str = None, aggregate_method: str = "mean"):
        self.name = name
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.aggregate_method = aggregate_method
        self.attention_pool = None 
        logger.info(f"Инициализация {name} экстрактора признаков на устройстве {self.device}")

    @abstractmethod
    def extract_single(self, data: DataModel) -> FeatureVector:
        """Извлекает признаки из одной модели"""
        pass
    

    def set_aggregate_method(self, method: str):
        if method not in ("mean", "max", "attention"):
            raise ValueError("aggregate_method должен быть 'mean' | 'max' | 'attention'")
        self.aggregate_method = method

    def set_attention_pool(self, attention_pool) -> None:
        """Установить attention-пулинг (объект с методом pool)"""
        self.attention_pool = attention_pool

    def _aggregate_features(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Агрегирует массив векторов признаков (для одной модели) в один вектор"""
        if feature_vectors.ndim != 2:
            feature_vectors = feature_vectors.reshape(-1, feature_vectors.shape[-1])
        if self.aggregate_method == 'mean':
            return np.mean(feature_vectors, axis=0)
        elif self.aggregate_method == 'max':
            return np.amax(feature_vectors, axis=0)
        elif self.aggregate_method == 'attention':
            if self.attention_pool is None:
                raise ValueError("Attention-пулинг не задан. Вызовите set_attention_pool(...)")
            return self.attention_pool.pool(feature_vectors)
        else:
            raise ValueError(f"Неизвестный метод агрегации: {self.aggregate_method}")

    def extract_batch(self, datasets: list[DataModel]) -> FeatureDataset:
        """Извлекает признаки из батча данных"""
        logger.info(f"Извлечение признаков с использованием {self.name} для {len(datasets)} моделей")

        features = []
        from tqdm import tqdm
        with tqdm(datasets, desc=f"Извлечение признаков {self.name}") as pbar:
            for data_model in pbar:
                try:
                    feature_vector = self.extract_single(data_model)
                    features.append(feature_vector)
                    pbar.set_postfix(extracted=len(features))
                except Exception as e:
                    logger.warning(f"Не удалось извлечь признаки для {data_model.model_id}: {e}")
                    continue
        
        feature_dataset = FeatureDataset(
            features=features,
            extractor_type=self.name
        )

        logger.success(f"Извлечено {len(features)} векторов признаков с использованием {self.name}")
        return feature_dataset

class FeatureIO:
    """Класс для ввода/вывода признаков"""
    
    @staticmethod
    def save_features(feature_dataset: FeatureDataset, filepath: Path) -> None:
        """Сохраняет признаки в pickle файл"""
        logger.info(f"Сохранение признаков {feature_dataset.extractor_type} в {filepath}")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(feature_dataset, f)

        logger.success(f"Признаки сохранены в {filepath}")
    
    @staticmethod
    def load_features(filepath: Path) -> FeatureDataset:
        """Загружает признаки из pickle файла"""
        logger.info(f"Загрузка признаков из {filepath}")
        
        with open(filepath, 'rb') as f:
            feature_dataset = pickle.load(f)

        logger.success(f"Признаки загружены из {filepath}")
        return feature_dataset

class FeatureExtractorFactory:
    """Фабрика для создания экстракторов признаков"""

    @staticmethod
    def create_extractor(extractor_type: ExtractorType) -> FeatureExtractor:
        """Создает экстрактор по типу"""
        if extractor_type == ExtractorType.CLIP_B32:
            from .clip import CLIPExtractor
            return CLIPExtractor()
        elif extractor_type == ExtractorType.CLIP_B16:
            from .clip import CLIPExtractor
            return CLIPExtractor(model="ViT-B/16")
        elif extractor_type == ExtractorType.CLIP_L14:
            from .clip import CLIPExtractor
            return CLIPExtractor(model="ViT-L/14")
        elif extractor_type == ExtractorType.DINO_BASE:
            from .dinov2 import DINOExtractor
            return DINOExtractor()
        elif extractor_type == ExtractorType.GOOGLE_B16_384:
            from .google_vit import GoogleVitExtractor
            return GoogleVitExtractor()
        elif extractor_type == ExtractorType.GOOGLE_B16_224:
            from .google_vit import GoogleVitExtractor
            return GoogleVitExtractor(model="google/vit-base-patch16-224")
        elif extractor_type == ExtractorType.BREP:
            from .brep import BrepExtractor
            return BrepExtractor()
        else:
            raise ValueError(f"Неизвестный тип экстрактора: {extractor_type}")



