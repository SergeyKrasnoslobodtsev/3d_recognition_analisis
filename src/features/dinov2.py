import numpy as np
from loguru import logger

from .extractor import FeatureExtractor, FeatureVector
from ..dataset import DataModel

class DINOExtractor(FeatureExtractor):
    """Экстрактор признаков DINO"""
    
    def __init__(self, model:str = None):
        super().__init__("DINO", model)

        logger.info("Loading DINO model...")
    
    def extract_single(self, data: DataModel) -> FeatureVector:
        """Извлекает DINO признаки из модели"""

        # Пока заглушка
        feature_vector = np.random.rand(768)  # DINO обычно 768-мерный
        
        return FeatureVector(
            model_id=data.model_id,
            feature_vector=feature_vector,
            extractor_type="dino",
            detail_type=data.detail_type
        )
    
    def get_feature_dimension(self) -> int:
        return 768