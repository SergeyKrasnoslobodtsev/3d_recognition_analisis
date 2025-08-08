from transformers import pipeline
import numpy as np

from .extractor import FeatureExtractor, FeatureVector
from ..dataset import DataModel

class GoogleVitExtractor(FeatureExtractor):
    """Экстрактор признаков для Google Vision Transformer"""

    def __init__(self, model: str = None):
        if model is None:
            model = "google/vit-base-patch16-384"
        
        super().__init__(name="GoogleVit", model=model)
        
        self.pipe = pipeline(task="image-feature-extraction", model_name=model, framework="pt", pool=True)
        
    def extract_single(self, data: DataModel) -> FeatureVector:
        """Извлекает признаки из одной модели"""
        raw_features = self.pipe(data.get_images())
        feature_vectors = np.array(raw_features).squeeze(axis=1)   
        aggregated_feature = self._aggregate_features(feature_vectors, method='mean')

        return FeatureVector(
            model_id=data.model_id,
            vector=aggregated_feature,
            label=data.detail_type
        )
