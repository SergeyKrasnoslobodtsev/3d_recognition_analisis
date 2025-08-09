from pathlib import Path

import typer

from ..config import INTERIM_DATA_DIR, RAW_DATA_DIR

from ..dataset import DatasetProcessor, DatasetIO, DatasetAnalyzer

app = typer.Typer()

@app.command()
def create_dataset(
    input_2d_path: Path = typer.Option(RAW_DATA_DIR / "2D", help="Путь к директории с 2D изображениями"),
    input_3d_path: Path = typer.Option(RAW_DATA_DIR / "3D", help="Путь к директории с 3D моделями"),
    output_path: Path = typer.Option(INTERIM_DATA_DIR / "dataset_metadata.pkl", help="Путь к выходному pickle файлу"),
    show_stats: bool = typer.Option(True, help="Показать статистику датасета после создания"),
):
    """Создает метаданные датасета в формате pickle из сырых 3D моделей и 2D изображений."""

    processor = DatasetProcessor()
    io_handler = DatasetIO()
    analyzer = DatasetAnalyzer()
    
    # Создаем датасет
    dataset = processor.create_dataset(input_3d_path, input_2d_path)
    
    # Сохраняем датасет
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io_handler.save_dataset_pickle(dataset, output_path)
    
    # Показываем статистику
    if show_stats:
        analyzer.print_dataset_stats(dataset)


@app.command()
def create_dataset_multiview(
    
    input_3d_path: Path = typer.Option(RAW_DATA_DIR / "3D", help="Путь к директории с 3D моделями"),
    output_2D_path: Path = typer.Option(RAW_DATA_DIR / "multiview", help="Путь к директории с 2D изображениями созданных из 3D моделей"),
    output_path: Path = typer.Option(INTERIM_DATA_DIR / "dataset_metadata.pkl", help="Путь к выходному pickle файлу"),
    show_stats: bool = typer.Option(True, help="Показать статистику датасета после создания"),
):
    """Создает метаданные датасета в формате pickle из сырых 3D моделей и 2D изображений."""

    processor = DatasetProcessor()
    io_handler = DatasetIO()
    analyzer = DatasetAnalyzer()
    
    # Создаем датасет
    dataset = processor.create_dataset_from_3d(input_3d_path, output_2D_path)
    
    # Сохраняем датасет
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io_handler.save_dataset_pickle(dataset, output_path)
    
    if show_stats:
        analyzer.print_dataset_stats(dataset)


if __name__ == "__main__":
    app()