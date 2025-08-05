import numpy as np
from loguru import logger

from .extractor import FeatureExtractor, FeatureVector
from ..dataset import DataModel

class BRepNetExtractor(FeatureExtractor):
    """Экстрактор признаков BRepNet"""
    
    def __init__(self):
        super().__init__("BRepNet")
        # Здесь инициализация BRepNet модели
        logger.info("Loading BRepNet model...")
    
    def extract_single(self, data: DataModel) -> FeatureVector:
        """Извлекает BRepNet признаки из модели"""
        # Здесь ваша логика извлечения BRepNet признаков
        # Пока заглушка
        feature_vector = np.random.rand(768)  # BRepNet обычно 768-мерный

        return FeatureVector(
            model_id=data.model_id,
            feature_vector=feature_vector,
            extractor_type="brepnet",
            detail_type=data.detail_type
        )
    
    def get_feature_dimension(self) -> int:
        return 768