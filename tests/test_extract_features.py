import numpy as np
import tests.conftest
from src.features.extractor import ExtractorType
from src.pipelines import ext_features_pipeline
from src.config import INTERIM_DATA_DIR, INTERIM_DATA_DIR

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from src.features.extractor import FeatureIO

PKL_BASE = INTERIM_DATA_DIR / "dataset_metadata.pkl"
PKL_MULTIVIEW = INTERIM_DATA_DIR / "dataset_metadata_multiview.pkl"
BREP = INTERIM_DATA_DIR / "brep_features.pkl"

def test_cuda():
    import torch
    print(torch.cuda.is_available())
    x = torch.rand(5, 3)
    print(x)

def test_print_list_extractors():
    ext_features_pipeline.list_extractors()


def test_ext_features_base():
    ext_features_pipeline.extract(
        dataset_path=PKL_BASE,
        extractor_type=ExtractorType.BREP,
        output_dir=INTERIM_DATA_DIR
    )

def test_ext_features_multiview():
    ext_features_pipeline.extract(
        dataset_path=PKL_MULTIVIEW,
        extractor_type=None,
        output_dir=INTERIM_DATA_DIR
    )

def test_train_features_brep():
    # 1. Загрузить извлеченные ранее признаки
    feature_dataset = FeatureIO.load_features(BREP)

    # 2. Преобразовать в массивы X и y
    X = np.array([fv.vector for fv in feature_dataset.features])
    y = np.array([fv.label for fv in feature_dataset.features])

    # 3. Разделить данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Создать и ОБУЧИТЬ скейлер ТОЛЬКО на обучающих данных
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 5. Применить обученный скейлер ко всем данным
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Обучить модель на нормализованных данных
    model = SVC()
    model.fit(X_train_scaled, y_train)

    # 7. Оценить модель
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Точность модели: {accuracy}")
