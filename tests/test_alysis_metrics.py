import tests.conftest
from src.features.extractor import ExtractorType
from src.pipelines import analitics_feature
from src.config import INTERIM_DATA_DIR, INTERIM_DATA_DIR

def test_analize_features():
    analitics_feature.analyze(
        features_path=None,
        extractor_type=None,
        features_dir=INTERIM_DATA_DIR,
        analyze_all=True,
        show_plots=False,
        save_plots=True
    )

def test_analize_compare():
    analitics_feature.compare(
        extractors=None,
        features_dir=INTERIM_DATA_DIR
    )