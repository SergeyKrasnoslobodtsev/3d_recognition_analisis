import numpy as np
from loguru import logger

from ..dataset import DataModel
from .extractor import FeatureExtractor, FeatureVector


from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                               GeomAbs_Sphere, GeomAbs_Torus)


class BrepExtractor(FeatureExtractor):
    
    def __init__(self, model: str = None):

        self.name = "BRep"
        self.model = model
        logger.info(f"Инициализация {self.name} экстрактора признаков")

    def extract_single(self, data: DataModel) -> FeatureVector:
        """
        Извлекает признаки для 3D-модели, усредняя векторы признаков со всех её граней.
        """
        reader = STEPControl_Reader()
        if not reader.ReadFile(data.model_path):
            raise RuntimeError(f"Не удалось прочитать файл: {data.model_path}")
            
        reader.TransferRoots()
        shape = reader.Shape(1)

        if shape.IsNull():
            raise ValueError("В файле не найдена геометрия (shape is null)")

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        all_face_features = []
        
        # Собираем векторы признаков с каждой грани
        while face_explorer.More():
            face = face_explorer.Current()
            # Используем ваш метод для получения признаков одной грани
            face_features = self.get_face_features(face)
            all_face_features.append(face_features)
            face_explorer.Next()
        
        # Усредняем признаки по всем граням, чтобы получить один вектор для модели
        if all_face_features:
            final_feature_vector = np.mean(all_face_features, axis=0, dtype=np.float32)
        else:
            # Если в модели нет граней, возвращаем нулевой вектор.
            # Длина 12 соответствует выходу get_face_features.
            final_feature_vector = np.zeros(12, dtype=np.float32)

        return FeatureVector(
            model_id=data.model_id,
            vector=final_feature_vector,
            label=data.detail_type
        )
    
    @staticmethod  
    def get_face_features(face) -> list[float]:
        # Эта функция остается без изменений
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        adaptor = BRepAdaptor_Surface(face, True)
        surface_type = adaptor.GetType()
        type_features = [0.0] * 6
        if surface_type == GeomAbs_Plane: type_features[0] = 1.0
        elif surface_type == GeomAbs_Cylinder: type_features[1] = 1.0
        elif surface_type == GeomAbs_Sphere: type_features[2] = 1.0
        elif surface_type == GeomAbs_Cone: type_features[3] = 1.0
        elif surface_type == GeomAbs_Torus: type_features[4] = 1.0
        else: type_features[5] = 1.0
        geom_params = [0.0] * 4
        if surface_type == GeomAbs_Cylinder: geom_params[0] = adaptor.Cylinder().Radius()
        elif surface_type == GeomAbs_Sphere: geom_params[0] = adaptor.Sphere().Radius()
        elif surface_type == GeomAbs_Cone:
            geom_params[0] = adaptor.Cone().RefRadius()
            geom_params[2] = adaptor.Cone().SemiAngle()
        elif surface_type == GeomAbs_Torus:
            geom_params[0] = adaptor.Torus().MajorRadius()
            geom_params[1] = adaptor.Torus().MinorRadius()
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        num_edges = 0
        while edge_explorer.More():
            num_edges += 1
            edge_explorer.Next()
        # Возвращаем список признаков 
        # площади, тип поверхности, геометрические параметры и количество рёбер
        return [area] + type_features + geom_params + [float(num_edges)]