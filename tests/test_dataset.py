import tests.conftest

from src.pipelines import data_pipeline
from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR

DATASET_2D = RAW_DATA_DIR / '2D'
DATASET_3D = RAW_DATA_DIR / '3D'
PKL_BASE = INTERIM_DATA_DIR / "dataset_metadata.pkl"
PKL_MULTIVIEW = INTERIM_DATA_DIR / "dataset_metadata_multiview.pkl"

def test_create_dataset():
    data_pipeline.create_dataset(
        input_2d_path=DATASET_2D,
        input_3d_path=DATASET_3D,
        output_path=PKL_BASE
    )

def test_create_dataset_from_3d():
    data_pipeline.create_dataset_multiview(
        input_3d_path=DATASET_3D,
        output_2D_path=RAW_DATA_DIR / "multiview",
        output_path=PKL_MULTIVIEW
    )