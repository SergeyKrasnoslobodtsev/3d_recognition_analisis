from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum

import pickle

from loguru import logger
import torch
from tqdm import tqdm

import numpy as np

from ..dataset import DataModel

class ExtractorType(str, Enum):
    CLIP = "clip"
    DINO = "dino"
    CNN_1D = "cnn_1d"
    BREPNET = "brepnet"

@dataclass
class FeatureVector:
    """Вектор признаков"""
    model_id: str
    feature_vector: np.ndarray
    extractor_type: str
    detail_type: str

@dataclass
class FeatureDataset:
    """Датасет признаков"""
    features: list[FeatureVector]
    extractor_type: str
    feature_dimension: int
    
    def __len__(self) -> int:
        return len(self.features)

class FeatureExtractor(ABC):
    """Абстрактный класс для извлечения признаков"""
    
    def __init__(self, name: str):
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Инициализация {name} экстрактора признаков на устройстве {self.device}")

    @abstractmethod
    def extract_single(self, data: DataModel) -> FeatureVector:
        """Извлекает признаки из одной модели"""
        pass
    
    @abstractmethod
    def get_feature_dimension(self) -> int:
        """Возвращает размерность извлекаемых признаков"""
        pass

    @staticmethod
    def _aggregate_features(feature_vectors: np.ndarray, method: str = 'mean') -> np.ndarray:
        """Агрегирует массив векторов признаков (для одной модели) в один вектор"""
        if method == 'mean':
            return np.mean(feature_vectors, axis=0)
        elif method == 'max':
            return np.max(feature_vectors, axis=0)
        else:
            raise ValueError(f"Неизвестный метод агрегации: {method}. Используйте 'mean' или 'max'.")

    def extract_batch(self, datasets: list[DataModel]) -> FeatureDataset:
        """Извлекает признаки из батча данных"""
        logger.info(f"Извлечение признаков с использованием {self.name} для {len(datasets)} моделей")

        features = []
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
            extractor_type=self.name,
            feature_dimension=self.get_feature_dimension()
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
        if extractor_type == ExtractorType.CLIP:
            from .clip import CLIPExtractor
            return CLIPExtractor()
        elif extractor_type == ExtractorType.DINO:
            from .dinov2 import DINOExtractor
            return DINOExtractor()
        elif extractor_type == ExtractorType.CNN_1D:
            from .cnn_1d import CNN1DExtractor
            return CNN1DExtractor()
        elif extractor_type == ExtractorType.BREPNET:
            from .brepnet import BRepNetExtractor
            return BRepNetExtractor()
        else:
            raise ValueError(f"Неизвестный тип экстрактора: {extractor_type}")



