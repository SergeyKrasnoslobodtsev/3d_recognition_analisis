from typing import Optional, List
from pathlib import Path
from loguru import logger
import pandas as pd
import typer

from ..features.extractor import ExtractorType, FeatureDataset, FeatureIO
from ..config import INTERIM_DATA_DIR, FIGURES_DIR, REPORTS_DIR
from ..plots import (
    plot_similarity_heatmap, 
    plot_matrix_error, 
    plot_clustering_tsne, 
    plot_clustering_umap
)
from ..metrics import FeatureMetrics

app = typer.Typer()


def analyze_features_single(
    feature_dataset: FeatureDataset,
    extractor_name: str,
    show_plots: bool = True,
    save_plots: bool = True
) -> None:
    """Анализирует признаки одного экстрактора"""
    
    logger.info(f"Анализ признаков {extractor_name}")
    logger.info(f"Количество образцов: {len(feature_dataset.features)}")
    
    if feature_dataset.features:
        logger.info(f"Размерность признаков: {feature_dataset.features[0].vector.shape}")
    
    # Извлекаем массивы признаков и лейблы из FeatureDataset
    features = [fv.vector for fv in feature_dataset.features]
    labels = [fv.label for fv in feature_dataset.features]
    
    logger.info(f"Уникальные классы: {set(labels)}")
    logger.info(f"Количество классов: {len(set(labels))}")
    
    # Вычисляем метрики
    metrics = FeatureMetrics(features, labels)
    logger.info(f"=== Метрики для {extractor_name} ===")
    metrics.print_all_metrics()
    
    # Определяем имя файла для сохранения графиков
    plot_name = extractor_name if save_plots else None
    
    # Строим графики
    if show_plots or save_plots:
        logger.info(f"Построение графиков для {extractor_name}")
        
        try:
            plot_similarity_heatmap(features, labels, plot_name)
        except Exception as e:
            logger.warning(f"Ошибка при построении heatmap: {e}")
        
        try:
            plot_matrix_error(features, labels, plot_name)
        except Exception as e:
            logger.warning(f"Ошибка при построении матрицы ошибок: {e}")
        
        try:
            plot_clustering_tsne(features, labels, plot_name)
        except Exception as e:
            logger.warning(f"Ошибка при построении t-SNE: {e}")
        
        try:
            plot_clustering_umap(features, labels, plot_name)
        except Exception as e:
            logger.warning(f"Ошибка при построении UMAP: {e}")
    
def analyze_features_multiple(
    feature_datasets: dict[str, FeatureDataset],
    show_plots: bool = True,
    save_plots: bool = True
) -> None:
    """Анализирует признаки нескольких экстракторов"""
    
    logger.info(f"Сравнительный анализ {len(feature_datasets)} экстракторов")
    
    # Анализируем каждый экстрактор отдельно
    for extractor_name, feature_dataset in feature_datasets.items():
        analyze_features_single(
            feature_dataset, 
            extractor_name, 
            show_plots, 
            save_plots
        )
        logger.info("=" * 50)


@app.command()
def analyze(
    features_path: Optional[Path] = typer.Option(
        None,
        help="Путь к конкретному файлу с признаками (например, google_b16_features.pkl)"
    ),
    extractor_type: Optional[ExtractorType] = typer.Option(
        None,
        help="Тип экстрактора для анализа (если не указан features_path)"
    ),
    features_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        help="Директория с файлами признаков"
    ),
    analyze_all: bool = typer.Option(
        False,
        "--all",
        help="Анализировать все найденные файлы признаков"
    ),
    show_plots: bool = typer.Option(
        True,
        "--show/--no-show",
        help="Показывать графики"
    ),
    save_plots: bool = typer.Option(
        True,
        "--save/--no-save", 
        help="Сохранять графики"
    )
):
    """Анализирует извлеченные признаки с построением метрик и графиков"""
    
    # Убедимся, что директория для графиков существует
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if features_path:
        # Анализ конкретного файла
        if not features_path.exists():
            logger.error(f"Файл не найден: {features_path}")
            raise typer.Exit(1)
            
        feature_dataset = FeatureIO.load_features(features_path)
        extractor_name = features_path.stem.replace('_features', '')
        
        analyze_features_single(feature_dataset, extractor_name, show_plots, save_plots)
        
    elif extractor_type:
        # Анализ конкретного экстрактора
        features_file = features_dir / f"{extractor_type.value}_features.pkl"
        if not features_file.exists():
            logger.error(f"Файл признаков не найден: {features_file}")
            logger.info("Сначала извлеките признаки с помощью: python -m src.pipelines.ext_features_pipeline extract")
            raise typer.Exit(1)
            
        feature_dataset = FeatureIO.load_features(features_file)
        analyze_features_single(feature_dataset, extractor_type.value, show_plots, save_plots)
        
    elif analyze_all:
        # Анализ всех найденных файлов признаков
        feature_files = list(features_dir.glob("*_features.pkl"))
        
        if not feature_files:
            logger.error(f"Файлы признаков не найдены в {features_dir}")
            logger.info("Сначала извлеките признаки с помощью: python -m src.pipelines.ext_features_pipeline extract")
            raise typer.Exit(1)
        
        feature_datasets = {}
        for file_path in feature_files:
            extractor_name = file_path.stem.replace('_features', '')
            feature_dataset = FeatureIO.load_features(file_path)
            feature_datasets[extractor_name] = feature_dataset
        
        analyze_features_multiple(feature_datasets, show_plots, save_plots)
        
    else:
        logger.error("Укажите один из параметров: --features-path, --extractor-type, или --all")
        raise typer.Exit(1)


@app.command()
def list_features():
    """Показывает список доступных файлов признаков"""
    
    feature_files = list(INTERIM_DATA_DIR.glob("*_features.pkl"))
    
    if feature_files:
        logger.info("Доступные файлы признаков:")
        for file_path in feature_files:
            try:
                feature_dataset = FeatureIO.load_features(file_path)
                extractor_name = file_path.stem.replace('_features', '')
                logger.info(f"  - {extractor_name}: {file_path}")
                logger.info(f"    Количество образцов: {len(feature_dataset.features)}")
                if feature_dataset.features:
                    logger.info(f"    Размерность: {feature_dataset.features[0].vector.shape}")
                    labels = [fv.label for fv in feature_dataset.features]
                    logger.info(f"    Классы: {len(set(labels))} уникальных")
            except Exception as e:
                logger.warning(f"Ошибка при загрузке {file_path}: {e}")
    else:
        logger.warning(f"Файлы признаков не найдены в {INTERIM_DATA_DIR}")
        logger.info("Сначала извлеките признаки с помощью: python -m src.pipelines.ext_features_pipeline extract")


@app.command()
def compare(
    extractors: List[ExtractorType] = typer.Option(
        None,
        help="Список экстракторов для сравнения"
    ),
    features_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        help="Директория с файлами признаков"
    )
):
    """Сравнивает метрики между разными экстракторами"""
    
    if not extractors:
        feature_files = list(features_dir.glob("*_features.pkl"))
        extractors = [file_path.stem.replace('_features', '') for file_path in feature_files]
    else:
        extractors = [ext.value for ext in extractors]
    
    results = []
    
    for extractor_name in extractors:
        features_file = features_dir / f"{extractor_name}_features.pkl"
        if not features_file.exists():
            continue
            
        try:
            feature_dataset = FeatureIO.load_features(features_file)
            features = [fv.vector for fv in feature_dataset.features]
            labels = [fv.label for fv in feature_dataset.features]
            
            metrics = FeatureMetrics(features, labels)
            result = metrics.get_metrics()
            result['extractor'] = extractor_name
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Ошибка при анализе {extractor_name}: {e}")
            continue
    
    if not results:
        logger.error("Нет данных для сравнения")
        return
    
    # Создаем DataFrame и выводим таблицу
    df = pd.DataFrame(results)
    df = df[['extractor', 
             'samples', 
             'classes', 
             'feature_dim', 
             'map', 
             'silhouette', 
             'intra_class_distance',
             'inter_class_distance',
             'separation_ratio']]

    # сохраняем отчет csv
    report_path = REPORTS_DIR / "feature_comparison_report.csv"
    df.to_csv(report_path, index=False)
    logger.success(f"Отчет сохранен: {report_path}")

    logger.info("Сравнение экстракторов:")
    logger.info(f"\n{df.to_string(index=False, float_format='%.3f')}")


if __name__ == "__main__":
    app()