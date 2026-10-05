from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "Artifacts"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

MODEL_PATH = ARTIFACTS_DIR / "titanic_rf_pipeline.joblib"
SUBMISSION_PATH = ARTIFACTS_DIR / "submission_from_main.csv"


# Reproducibility
RANDOM_STATE = 43


# Cross-validation
N_SPLITS = 5
SCORING = "accuracy"


# Model
MODEL_NAME = "random_forest"


# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
}
