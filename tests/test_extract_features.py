import tests.conftest
from src.features.extractor import ExtractorType
from src.pipelines import ext_features_pipeline
from src.config import INTERIM_DATA_DIR, INTERIM_DATA_DIR

PKL_BASE = INTERIM_DATA_DIR / "dataset_metadata.pkl"
PKL_MULTIVIEW = INTERIM_DATA_DIR / "dataset_metadata_multiview.pkl"

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
        extractor_type=None,
        output_dir=INTERIM_DATA_DIR
    )

def test_ext_features_multiview():
    ext_features_pipeline.extract(
        dataset_path=PKL_MULTIVIEW,
        extractor_type=None,
        output_dir=INTERIM_DATA_DIR
    )
