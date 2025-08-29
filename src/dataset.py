from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
from typing import List
from loguru import logger
from tqdm import tqdm
from PIL import Image
import pyvista as pv
import trimesh
import numpy as np

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
    class_name: str

    def get_pil_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")

@dataclass
class DataModel:
    model_id: str
    class_name: str
    model_path: str
    image_paths: List[ImageData] = field(default_factory=list)

    def add_image_data(self, image_data: ImageData) -> None:
        self.image_paths.append(image_data)

    def get_images(self) -> List[Image.Image]:
        return [img_data.get_pil_image() for img_data in self.image_paths]

class DatasetProcessor:
    def __init__(self, 
                 models_dir: Path, 
                 images_dir: Path, 
                 num_views: int = 36, 
                 size: int = 512,
                 mode:str = 'projection'):
        
        self.models_dir = models_dir.resolve()
        self.images_dir = images_dir.resolve()
        self.num_views = num_views
        self.size = size
        self.mode = mode

    def create_dataset(self) -> List[DataModel]:
        dataset = []

        for class_name, stp_file in tqdm(self._get_files(), desc="Обработка моделей"):
            model_id = stp_file.stem
            data_model = DataModel(
                model_id=model_id,
                class_name=class_name,
                model_path=str(stp_file.resolve())
            )
            output_dir = self.images_dir / class_name / model_id

            if self.mode == 'projection':
                self._render_model_views_types(stp_file, output_dir, size=self.size)
                self._link_rendered_images(data_model, output_dir, class_name)
            elif self.mode == 'rotation':
                self._render_model_views(stp_file, output_dir, size=self.size)
                self._link_rendered_images(data_model, output_dir, class_name)
            elif self.mode == 'technical':
                # удалим в названии output_dir .prt в конце если есть
                if output_dir.suffix == '.prt':
                    output_dir = output_dir.with_suffix('')

                self._link_rendered_images(data_model, output_dir, class_name)
            
            dataset.append(data_model)
        logger.success(f"Датасет сформирован: {len(dataset)} моделей.")
        return dataset
   
    def _get_files(self):
        all_files: list[tuple[str, Path]] = []
        for class_dir in self.models_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for stp_file in class_dir.glob("*.stp"):
                    all_files.append((class_name, stp_file))
        return all_files

    def _render_model_views_types(self, model_path: Path, output_dir: Path, size: int = 512):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            mesh_trimesh = trimesh.load(str(model_path), force='mesh')
            mesh_trimesh.apply_translation(-mesh_trimesh.center_mass)
            mesh = pv.wrap(mesh_trimesh)
            plotter = pv.Plotter(off_screen=True, window_size=[size, size])
            plotter.set_background('white')
            plotter.add_mesh(mesh, color='gray', smooth_shading=True)
            plotter.enable_anti_aliasing()
            plotter.reset_camera(bounds=mesh.bounds)
            plotter.show(auto_close=False, interactive=False)
            cx, cy, cz = mesh.center
            r = max(1e-6, 1.75 * np.linalg.norm(mesh.length))
            center = np.array([cx, cy, cz], dtype=float)
            offset = r * 0.1
            # Определяем позиции камер для каждого ModelType
            camera_positions = {
                ModelType.FRONT:  np.array([0, -r, 0]),
                ModelType.BACK:   np.array([0, r, 0]),
                ModelType.LEFT:   np.array([-r, 0, 0]),
                ModelType.RIGHT:  np.array([r, 0, 0]),
                ModelType.TOP:    np.array([0, 0, r + offset]),
                ModelType.BOTTOM: np.array([0, 0, -r - offset]),
                ModelType.ISOMETRIC: np.array([r, r, r]),
                ModelType.TRIMETRIC: np.array([r, r, -r]),
            }
            for view_name, pos in camera_positions.items():
                cam_pos = center + pos
                cam = plotter.camera
                cam.position = tuple(cam_pos.tolist())
                cam.focal_point = (cx, cy, cz)
                if view_name == ModelType.TOP:
                    cam.up = (0, 1, 0)
                elif view_name == ModelType.BOTTOM:
                    cam.up = (0, -1, 0)
                else:
                    cam.up = (0, 0, 1)
                plotter.render()
                output_path = output_dir / f"{view_name}.png"
                plotter.screenshot(output_path)
            plotter.close()
        except Exception as e:
            logger.error(f"Ошибка рендера {model_path}: {e}")

    def _render_model_views(self, model_path: Path, output_dir: Path, size: int = 512):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            mesh_trimesh = trimesh.load(str(model_path), force='mesh')
            mesh_trimesh.apply_translation(-mesh_trimesh.center_mass)
            mesh = pv.wrap(mesh_trimesh)

            plotter = pv.Plotter(off_screen=True, window_size=[size, size])
            plotter.set_background('white')
            plotter.add_mesh(mesh, color='gray', smooth_shading=True)
            plotter.enable_anti_aliasing()
            plotter.reset_camera(bounds=mesh.bounds)
            plotter.show(auto_close=False, interactive=False)

            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
            r = max(1e-6, 1.75 * diag)
            cx, cy, cz = mesh.center

            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            N = max(1, int(self.num_views))
            center = np.array([cx, cy, cz], dtype=float)

            def safe_up(cam_pos: np.ndarray, center: np.ndarray) -> np.ndarray:
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

            for i in range(N):
                z = 1.0 - 2.0 * ((i + 0.5) / N)
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
            logger.error(f"Ошибка рендера {model_path}: {e}")

    def _link_rendered_images(self, data_model: DataModel, images_dir: Path, class_name: str = ""):
        exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in exts:
            image_files.extend(images_dir.glob(ext))
        for image_file in image_files:
            image_data = ImageData(
                image_id=f"{data_model.model_id}_{image_file.stem}",
                image_path=str(image_file.resolve()),
                class_name=class_name
            )
            data_model.add_image_data(image_data)

class DatasetIO:
    @staticmethod
    def save_dataset_pickle(dataset: List[DataModel], filepath: Path) -> None:
        logger.info(f"Сохраняем датасет в {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        logger.success(f"Датасет сохранен: {filepath}")

    @staticmethod
    def load_dataset_pickle(filepath: Path) -> List[DataModel]:
        logger.info(f"Загрузка датасета из {filepath}")
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        logger.success(f"Датасет загружен: {filepath}")
        return dataset

class DatasetAnalyzer:
    @staticmethod
    def print_dataset_stats(dataset: List[DataModel]) -> None:
        total_models = len(dataset)
        total_images = sum(len(model.image_paths) for model in dataset)
        models_with_images = sum(1 for model in dataset if model.image_paths)
        logger.info(f"Всего моделей: {total_models}")
        logger.info(f"Всего изображений: {total_images}")
        logger.info(f"Моделей с изображениями: {models_with_images}")
        logger.info(f"Моделей без изображений: {total_models - models_with_images}")


from torch.utils.data import Dataset

class ModelImageDataset(Dataset):
    def __init__(self, data_models:list[DataModel], transform=None):
        self.samples = []
        for model in data_models:
            for img_data in model.image_paths:
                self.samples.append((img_data.image_path, img_data.class_name, img_data.image_id))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, id_x = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, id_x