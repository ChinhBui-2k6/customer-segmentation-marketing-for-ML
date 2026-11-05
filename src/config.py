"""Cau hinh dung chung cho toan bo pipeline Machine Learning.

Moi module trong src/ va cac notebook nen import tu file nay thay vi
hard-code duong dan hay tham so, de dam bao tinh nhat quan va tai lap.
"""

from pathlib import Path

# Duong dan (tu dong tinh tu vi tri file nay: src/config.py -> goc project)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_PREPROCESSING_DIR = FIGURES_DIR / "preprocessing"
FIGURES_DIM_REDUCTION_DIR = FIGURES_DIR / "dim_reduction"
FIGURES_CLUSTERING_DIR = FIGURES_DIR / "clustering"
FIGURES_CLASSIFICATION_DIR = FIGURES_DIR / "classification"
FIGURES_REGRESSION_DIR = FIGURES_DIR / "regression"

REPORT_DIR = PROJECT_ROOT / "report"
SLIDES_DIR = PROJECT_ROOT / "slides"

# Tai lap ket qua
RANDOM_SEED = 42

# Tham so dung chung
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
EXPERIMENT_VALIDATION_SIZES = (0.2, 0.3, 0.4)
DEFAULT_N_CLUSTERS = 4
KMEANS_N_INIT = 10
PCA_N_COMPONENTS = 2
FIGURE_DPI = 150
FIGURE_FORMAT = "png"

_OUTPUT_DIRS = (
    PROCESSED_DATA_DIR,
    FIGURES_PREPROCESSING_DIR,
    FIGURES_DIM_REDUCTION_DIR,
    FIGURES_CLUSTERING_DIR,
    FIGURES_CLASSIFICATION_DIR,
    FIGURES_REGRESSION_DIR,
    TABLES_DIR,
    REPORT_DIR,
    SLIDES_DIR,
)


def ensure_dirs() -> None:
    """Tao cac thu muc dau ra neu chua ton tai."""
    for directory in _OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
