import numpy as np
from loguru import logger

from .extractor import FeatureExtractor, FeatureVector
from ..dataset import DataModel

class CNN1DExtractor(FeatureExtractor):
    """Экстрактор признаков CNN 1D"""
    
    def __init__(self):
        super().__init__("CNN 1D")
        # Здесь инициализация CNN 1D модели
        logger.info("Loading CNN 1D model...")
    
    def extract_single(self, data: DataModel) -> FeatureVector:
        """Извлекает CNN 1D признаки из модели"""
        # Здесь ваша логика извлечения CNN 1D признаков
        # Пока заглушка
        feature_vector = np.random.rand(768)  # CNN 1D обычно 768-мерный

        return FeatureVector(
            model_id=data.model_id,
            feature_vector=feature_vector,
            extractor_type="cnn_1d",
            detail_type=data.detail_type
        )
    
    def get_feature_dimension(self) -> int:
        return 768