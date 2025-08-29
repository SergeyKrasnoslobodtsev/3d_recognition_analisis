from pathlib import Path

import typer

from ..config import INTERIM_DATA_DIR, RAW_DATA_DIR

from ..dataset import DatasetProcessor, DatasetIO, DatasetAnalyzer

app = typer.Typer()
# Комманда для запуска python -m src.pipelines.build_dataset
@app.command()
def run(
    models_dir: Path = typer.Option(RAW_DATA_DIR / "OLD3D", help="Путь к директории с 3D моделями"),
    images_dir: Path = typer.Option(RAW_DATA_DIR / "2D" / "v1", help="Путь к директории с 2D изображениями"),
    output_path: Path = typer.Option(INTERIM_DATA_DIR / "dataset_metadata_v1.pkl", help="Путь к выходному pickle файлу"),
    num_views: int = typer.Option(36, help="Количество видов для рендеринга"),
    img_size: int = typer.Option(512, help="Размер изображений для рендеринга"),
    mode: str = typer.Option("technical", help="Режим рендеринга: projection, rotation или technical"),
    show_stats: bool = typer.Option(True, help="Показать статистику датасета после создания"),
):
    """Генерирует pickle-файл с метаданными датасета, рендерит изображения из 3D моделей и сохраняет их в отдельную папку."""

    processor = DatasetProcessor(models_dir, images_dir, num_views=num_views, size=img_size, mode=mode)
    dataset = processor.create_dataset()
    DatasetIO.save_dataset_pickle(dataset, output_path)

    if show_stats:
        DatasetAnalyzer.print_dataset_stats(dataset)

if __name__ == "__main__":
    app()