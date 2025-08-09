import tests.conftest

from src.pipelines import ext_features_pipeline
from src.config import INTERIM_DATA_DIR, INTERIM_DATA_DIR

PKL_BASE = INTERIM_DATA_DIR / "dataset_metadata.pkl"
PKL_MULTIVIEW = INTERIM_DATA_DIR / "dataset_metadata_multiview.pkl"

def test_print_list_extractors():

    ext_features_pipeline.list_extractors()


def test_ext_features_base():
    ext_features_pipeline.extract(
        dataset_path=PKL_BASE,
        extractor_type='google_b16',
        output_dir=INTERIM_DATA_DIR
    )


def test_cuda():
    import torch
    print(torch.cuda.is_available())