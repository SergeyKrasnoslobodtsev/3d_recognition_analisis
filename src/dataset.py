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

import pyvista as pv
import trimesh

import numpy as np

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
    RENDERED_VIEW = 'rendered_view'


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
    
    def get_images(self) -> list[Image.Image]:
        """Возвращает список PIL изображений для данной модели"""
        images = []
        for img_data in self.image_paths:
            images.append(img_data.get_pil_image())
        return images

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

    def _render_model_views(self, model_path: Path, output_dir: Path, num_views: int = 36):
        """
        Рендерит виды модели.
        coverage:
          - "ring"  — орбита по одной широте (старое поведение)
          - "sphere"— равномерно по сфере (со всех сторон)
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            mesh_trimesh = trimesh.load(str(model_path), force='mesh')
            mesh_trimesh.apply_translation(-mesh_trimesh.center_mass)
            mesh = pv.wrap(mesh_trimesh)

            plotter = pv.Plotter(off_screen=True, window_size=[512, 512])
            plotter.set_background('black')
            plotter.add_mesh(mesh, color='white', smooth_shading=True)
            plotter.enable_anti_aliasing()
            plotter.reset_camera(bounds=mesh.bounds)
            # Прогрев рендера
            plotter.show(auto_close=False, interactive=False)

            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
            r = max(1e-6, 1.75 * diag)
            cx, cy, cz = mesh.center

            def safe_up(cam_pos: np.ndarray, center: np.ndarray) -> np.ndarray:
                # Строим up, перпендикулярный лучу взгляда, избегая вырождения у полюсов
                d = center - cam_pos
                d /= (np.linalg.norm(d) + 1e-12)
                ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
                right = np.cross(ref, d)
                if np.linalg.norm(right) < 1e-6:
                    ref = np.array([1.0, 0.0, 0.0])
                    right = np.cross(ref, d)
                right /= (np.linalg.norm(right) + 1e-12)
                up = np.cross(d, right)
                up /= (np.linalg.norm(up) + 1e-12)
                return up

            # Полное покрытие сферы: Fibonacci sphere (равномерные направления)
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            N = max(1, int(num_views))
            center = np.array([cx, cy, cz], dtype=float)

            for i in range(N):
                z = 1.0 - 2.0 * ((i + 0.5) / N)            # [-1, 1]
                rho = np.sqrt(max(0.0, 1.0 - z * z))
                theta = i * golden_angle
                nx = np.cos(theta) * rho
                ny = np.sin(theta) * rho
                nz = z
                pos = center + r * np.array([nx, ny, nz], dtype=float)

                up = safe_up(pos, center)

                cam = plotter.camera
                cam.position = tuple(pos.tolist())
                cam.focal_point = (cx, cy, cz)
                cam.up = tuple(up.tolist())
                plotter.render()

                output_path = output_dir / f"frame_{i:03d}.png"
                plotter.screenshot(output_path)

            plotter.close()
        except Exception as e:
            logger.error(f"Не удалось отрендерить модель {model_path.resolve()}: {e}")

    def _link_rendered_images(self, data_model: DataModel, rendered_views_dir: Path):
        """Находит отрендеренные изображения и добавляет их в DataModel."""
        model_name_base = self.normalize_name(Path(data_model.model_path).name)
        # Путь к папке с изображениями для конкретной модели
        images_dir = rendered_views_dir / data_model.detail_type / model_name_base
        
        if not images_dir.exists():
            return

        for image_file in images_dir.glob('*.png'):
            image_data = ImageData(
                image_id=f"{data_model.model_id}_{image_file.stem}",
                image_path=str(image_file),
                model_type=ModelType.RENDERED_VIEW.value
            )
            data_model.add_image_data(image_data)

    def create_dataset_from_3d(self, dir_3d: Path, rendered_views_dir: Path, num_views: int = 36) -> list[DataModel]:
        """
        Создает датасет, генерируя 2D изображения из 3D моделей.
        """
        logger.info("Создание датасета из 3D моделей...")
        
        structure_3d = self.get_directory_structure(dir_3d)
        
        # 1. Обрабатываем 3D модели для создания базовых DataModel
        models_tuples = self.process_3d_files(structure_3d, dir_3d)
        dataset = [model for _, model in models_tuples]

        # 2. Рендерим каждую модель и связываем изображения
        with tqdm(dataset, desc="Рендеринг и связывание видов") as pbar:
            for data_model in pbar:
                model_name_base = self.normalize_name(Path(data_model.model_path).name)
                output_dir = rendered_views_dir / data_model.detail_type / model_name_base
                
                # Рендерим виды
                self._render_model_views(Path(data_model.model_path), output_dir, num_views)
                
                # Связываем созданные изображения с моделью
                self._link_rendered_images(data_model, rendered_views_dir)
                
                pbar.set_postfix(images=len(data_model.image_paths))
                # break # хардкординг

        logger.success(f"Датасет успешно создан. Моделей: {len(dataset)}")
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


