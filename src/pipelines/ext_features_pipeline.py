
from typing import Optional
from pathlib import Path
from loguru import logger


from ..features.extractor import ExtractorType, FeatureExtractorFactory, FeatureDataset, FeatureIO
from ..config import INTERIM_DATA_DIR
from ..dataset import DataModel, DatasetIO

import typer

app = typer.Typer()


def extract_features_single(
    extractor_type: ExtractorType,
    dataset: list[DataModel],
    output_dir: Path = INTERIM_DATA_DIR
) -> FeatureDataset:
    """Извлекает признаки одним экстрактором"""
    
    extractor = FeatureExtractorFactory.create_extractor(extractor_type)
    feature_dataset = extractor.extract_batch(dataset)
    
    # Сохраняем результат
    output_path = output_dir / f"{extractor_type.value}_features.pkl"
    FeatureIO.save_features(feature_dataset, output_path)
    
    return feature_dataset

def extract_features_all(
    dataset: list[DataModel],
    extractors: list[ExtractorType],
    output_dir: Path = INTERIM_DATA_DIR
) -> dict[str, FeatureDataset]:
    """Извлекает признаки всеми указанными экстракторами"""
    
    results = {}
    for extractor_type in extractors:
        logger.info(f"Начало извлечения признаков с использованием {extractor_type.value}")
        feature_dataset = extract_features_single(extractor_type, dataset, output_dir)
        results[extractor_type.value] = feature_dataset

    logger.success(f"Завершено извлечение признаков с использованием {len(extractors)} экстракторов")
    return results


@app.command()
def extract(
    dataset_path: Path = typer.Option(
        INTERIM_DATA_DIR / "dataset_metadata.pkl", 
        help="Путь к файлу с метаданными датасета"
    ),
    extractor_type: Optional[ExtractorType] = typer.Option(
        None, 
        help="Конкретный экстрактор для использования (если не указан, будут использованы все)"
    ),
    output_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        help="Путь к директории для сохранения файлов признаков"
    ),
):
    """Извлекает признаки из датасета с использованием указанных экстракторов"""
    
    # Загружаем датасет
    dataset = DatasetIO.load_dataset_pickle(dataset_path)
    
    if extractor_type:
        # Извлекаем признаки одним экстрактором
        extract_features_single(extractor_type, dataset, output_dir)
    else:
        # Извлекаем признаки всеми экстракторами
        all_extractors = list(ExtractorType)
        extract_features_all(dataset, all_extractors, output_dir)

@app.command()
def list_extractors():
    """Список доступных экстракторов признаков"""
    logger.info("Доступные экстракторы признаков:")
    for extractor in ExtractorType:
        logger.info(f"  - {extractor.value}")

if __name__ == "__main__":
    app()