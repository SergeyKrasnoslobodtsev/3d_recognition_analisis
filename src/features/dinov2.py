from transformers import pipeline
import numpy as np

from .extractor import FeatureExtractor, FeatureVector
from ..dataset import DataModel

from loguru import logger

class DINOExtractor(FeatureExtractor):
    """Экстрактор признаков Facebook DINO"""
    
    def __init__(self, model: str = None):
        if model is None:
            model = "facebook/dinov2-base"

        super().__init__(name="DINOv2", model=model)

        self.pipe = pipeline(task="image-feature-extraction", model_name=model, framework="pt", pool=True)
    
    def _to_vector(self, out) -> np.ndarray:
        arr = np.array(out)
        if arr.ndim == 1:
            return arr.astype(np.float32)
        if arr.ndim == 2:
            return arr.mean(axis=0).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr.squeeze(0).mean(axis=0).astype(np.float32)
        return arr.reshape(-1).astype(np.float32)
       
    def extract_single(self, data: DataModel) -> FeatureVector:
        """Извлекает признаки из одной модели"""
        images = data.get_images()
        if not images:
            logger.warning(f"DINOv2: нет изображений для {data.model_id}")
            return None

        raw_list = self.pipe(images)
        per_view = [self._to_vector(out) for out in raw_list]
        feature_vectors = np.stack(per_view, axis=0)
        aggregated_feature = self._aggregate_features(feature_vectors)

        return FeatureVector(
            model_id=data.model_id,
            vector=aggregated_feature,
            label=data.detail_type
        )
    