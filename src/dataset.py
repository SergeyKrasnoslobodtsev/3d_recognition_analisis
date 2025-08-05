from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import pickle
import re
from typing import Optional

from loguru import logger
from tqdm import tqdm
import typer
from PIL import Image

app = typer.Typer()


class ModelType(str, Enum):
    BACK = 'back'
    BOTTOM = 'bottom'
    FRONT = 'front'
    LEFT = 'left'
    RIGHT = 'right'
    TOP = 'top'
    ISOMETRIC = 'isometric'
    TRIMETRIC = 'trimetric'


@dataclass
class ImageData:
    image_id: str
    image_path: str
    model_type: str

    def get_pil_image(self):
        """Загружает изображение как PIL Image"""
        return Image.open(self.image_path).convert("RGB")


@dataclass
class DataModel:
    model_id: str
    model_path: str
    detail_type: str
    image_paths: list[ImageData] = field(default_factory=list)

    def add_image_data(self, image_data: ImageData) -> None:
        self.image_paths.append(image_data)
    


class DatasetProcessor:
    """Обработка raw данных для создания датасета"""
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Нормализует имя для сравнения (убирает расширения и лишние символы)"""
        return re.sub(r'\.(prt\.stp|stp|jpg)$', '', name)

    @staticmethod
    def extract_model_type_from_filename(filename: str) -> Optional[ModelType]:
        """Извлекает тип модели из имени файла"""
        filename_lower = filename.lower()
        
        for model_type in ModelType:
            if f'_{model_type.value}.' in filename_lower:
                return model_type
        
        return None

    @staticmethod
    def get_directory_structure(directory: Path) -> dict[str, list[str]]:
        """Получает структуру директорий и файлов в виде словаря"""
        structure = {}
        for root, dirs, files in os.walk(directory):
            relative_root = os.path.relpath(root, directory)
            structure[relative_root] = files
        return structure

    def process_3d_files(self, structure_3d: dict, dir_3d: Path) -> list[tuple[str, DataModel]]:
        """Обрабатывает 3D файлы и создает базовые DataModel объекты"""
        models = []
        
        # Подсчитываем общее количество .stp файлов для прогресс-бара
        total_stp_files = sum(
            len([f for f in files if f.endswith('.stp')]) 
            for subdir, files in structure_3d.items() 
            if subdir != '.'
        )

        with tqdm(total=total_stp_files, desc="Обработка 3D моделей") as pbar:
            for subdir_3d, files_3d in structure_3d.items():
                if subdir_3d == '.':
                    continue
                    
                detail_type = subdir_3d.split(os.path.sep)[0]
                
                for file_3d in files_3d:
                    if not file_3d.endswith('.stp'):
                        continue
                    
                    model_path = dir_3d / subdir_3d / file_3d
                    model_id = self.normalize_name(file_3d)
                    
                    data_model = DataModel(
                        model_id=model_id,
                        model_path=str(model_path),
                        detail_type=detail_type
                    )
                    
                    models.append((subdir_3d, data_model))
                    pbar.update(1)

        logger.info(f"Обработано {len(models)} 3D моделей")
        return models

    def process_2d_images(self, models: list[tuple[str, DataModel]], 
                         structure_2d: dict, dir_2d: Path) -> list[DataModel]:
        """Обрабатывает 2D изображения и связывает их с 3D моделями"""
        dataset = []
        
        with tqdm(models, desc="Связывание 2D изображений") as pbar:
            for subdir_3d, data_model in pbar:
                model_name_base = self.normalize_name(Path(data_model.model_path).name)
                
                # Проверяем соответствующую папку в 2D
                if subdir_3d in structure_2d:
                    self._process_model_images(data_model, model_name_base, 
                                             subdir_3d, dir_2d)
                
                dataset.append(data_model)
                pbar.set_postfix(images=len(data_model.image_paths))
        
        return dataset

    def _process_model_images(self, data_model: DataModel, model_name_base: str,
                            subdir_2d: str, dir_2d: Path) -> None:
        """Обрабатывает изображения для конкретной модели"""
        subdir_path = dir_2d / subdir_2d
        
        for item in os.listdir(subdir_path):
            item_path = subdir_path / item
            
            if not item_path.is_dir():
                continue
                
            item_normalized = self.normalize_name(item)
            
            if item_normalized == model_name_base:
                self._add_images_to_model(data_model, item_path)

    def _add_images_to_model(self, data_model: DataModel, images_dir: Path) -> None:
        """Добавляет изображения к модели"""
        for image_file in os.listdir(images_dir):
            if not image_file.endswith('.jpg'):
                continue
                
            model_type = self.extract_model_type_from_filename(image_file)
            if not model_type:
                continue
                
            image_path = images_dir / image_file
            image_id = f"{data_model.model_id}_{model_type.value}"
            
            image_data = ImageData(
                image_id=image_id,
                image_path=str(image_path),
                model_type=model_type.value
            )
            
            data_model.add_image_data(image_data)

    def create_dataset(self, dir_3d: Path, dir_2d: Path) -> list[DataModel]:
        """Создает датасет, объединяя 3D модели с соответствующими 2D изображениями"""
        logger.info("Создание датасета...")
        
        structure_3d = self.get_directory_structure(dir_3d)
        structure_2d = self.get_directory_structure(dir_2d)
        
        # Обрабатываем 3D модели
        models = self.process_3d_files(structure_3d, dir_3d)
        
        # Связываем с 2D изображениями
        dataset = self.process_2d_images(models, structure_2d, dir_2d)

        logger.success(f"Датасет успешно создан {len(dataset)} моделей")
        return dataset


class DatasetIO:
    """Класс для ввода/вывода датасета"""
    
    @staticmethod
    def save_dataset_pickle(dataset: list[DataModel], filepath: Path) -> None:
        """Сохраняет датасет в pickle файл"""
        logger.info(f"Сохранение датасета в {filepath}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.success(f"Датасет сохранен в {filepath}")

    @staticmethod
    def load_dataset_pickle(filepath: Path) -> list[DataModel]:
        """Загружает датасет из pickle файла"""
        logger.info(f"Загрузка датасета из {filepath}")
        
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        
        logger.success(f"Датасет загружен из {filepath}")
        return dataset


class DatasetAnalyzer:
    """Класс для анализа датасета"""
    
    @staticmethod
    def print_dataset_stats(dataset: list[DataModel]) -> None:
        """Выводит статистику по датасету"""
        total_models = len(dataset)
        total_images = sum(len(model.image_paths) for model in dataset)
        models_with_images = sum(1 for model in dataset if model.image_paths)

        logger.info("Статистика по датасету:")
        logger.info(f"Всего 3D моделей: {total_models}")
        logger.info(f"Всего 2D изображений: {total_images}")
        logger.info(f"Моделей с изображениями: {models_with_images}")
        logger.info(f"Моделей без изображений: {total_models - models_with_images}")

        # Статистика по типам изображений
        type_counts = {}
        for model in dataset:
            for image in model.image_paths:
                type_counts[image.model_type] = type_counts.get(image.model_type, 0) + 1

        logger.info("Распределение типов изображений:")
        for model_type, count in sorted(type_counts.items()):
            logger.info(f"  {model_type}: {count}")


