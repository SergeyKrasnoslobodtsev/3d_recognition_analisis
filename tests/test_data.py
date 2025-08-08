import tests.conftest

from src.pipelines import data_pipeline
from src.pipelines import ext_features_pipeline
from loguru import logger


def test_create_dataset():
    data_pipeline.main()

def test_list_extractors():
    ext_features_pipeline.list_extractors()

def test_create_ext_features():
    ext_features_pipeline.extract(extractor_type='google_b16')


