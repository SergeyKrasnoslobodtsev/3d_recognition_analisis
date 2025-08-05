import numpy as np
from loguru import logger
import clip
import torch
from .extractor import FeatureExtractor, FeatureVector
from ..dataset import DataModel

class CLIPExtractor(FeatureExtractor):
    """Экстрактор признаков CLIP OpenAI"""
    
    def __init__(self, model:str = None):
        """Инициализация экстрактора CLIP"""
        super().__init__("CLIP", model)
        self.model, self.preprocess = clip.load(model, device=self.device)
        logger.success("Загрузка модели CLIP OpenAI завершена")
    
    def extract_single(self, data: DataModel) -> FeatureVector:
        """Извлекает CLIP признаки из модели"""
        try:
            model_features = []
            for img_data in data.image_paths:
                image = img_data.get_pil_image()
                if image is None:
                    logger.error(f"Не удалось загрузить изображение для {data.model_id}")
                    continue

                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    features = self.model.encode_image(image_tensor)
                    features /= features.norm(dim=-1, keepdim=True)
                    model_features.append(features[0].cpu().numpy())
            
            aggregated_vector = self._aggregate_features(np.array(model_features))
            
            return FeatureVector(
                        model_id=data.model_id,
                        feature_vector=aggregated_vector,
                        extractor_type="clip",
                        detail_type=data.detail_type
                    )
        
        except Exception as e:
            logger.error(f"Не удалось извлечь признаки для {data.model_id}: {e}")
            return None 
    
    def get_feature_dimension(self) -> int:
        return 512