
from dataclasses import dataclass, field
import pickle
from loguru import logger

@dataclass
class FeatureData:
    lables: list[str] = field(default_factory=list)
    features: list[any] = field(default_factory=list)

    def __dir__(self):
        return ["labels", "features"]


class FeatureIO:
    """Класс для работы с признаками"""
    
    @staticmethod
    def save_features(feature_data: FeatureData, file_path: str) -> None:
        """Сохраняет признаки в файл"""
        with open(file_path, 'wb') as f:
            pickle.dump(feature_data, f)
        logger.success(f"Признаки сохранены в {file_path}")

    @staticmethod
    def load_features(file_path: str) -> FeatureData:
        """Загружает признаки из файла"""
        with open(file_path, 'rb') as f:
            feature_data = pickle.load(f)
        logger.success(f"Признаки загружены из {file_path}")
        return feature_data

    @staticmethod
    def log_feature_info(feature_data: FeatureData) -> None:
        """Логирует информацию о признаках"""
        logger.info(f"Количество классов: {len(set(feature_data.lables))}")
        logger.info(f"Количество признаков: {len(feature_data.features)}")
        logger.info(f'Размерность признаков: {len(feature_data.features[0])}')
