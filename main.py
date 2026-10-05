import joblib
import pandas as pd

from Configs.config import (
    MODEL_NAME,
    MODEL_PATH,
    N_SPLITS,
    RANDOM_STATE,
    RF_PARAMS,
    SCORING,
    SUBMISSION_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
from src.inference import predict_survival
from src.training import build_pipeline, evaluate_pipeline


def main() -> None:
    """Train, evaluate and run the Titanic production pipeline."""

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X = train.drop(columns=["Survived"])
    y = train["Survived"]

    if MODEL_NAME != "random_forest":
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

    pipeline = build_pipeline(
        random_state=RANDOM_STATE,
        **RF_PARAMS,
    )

    cv_scores = evaluate_pipeline(
        pipeline=pipeline,
        X=X,
        y=y,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        scoring=SCORING,
    )

    print(f"Model: {MODEL_NAME}")
    print(f"CV scoring: {SCORING}")
    print(f"CV scores: {cv_scores}")
    print(
        f"CV mean: {cv_scores.mean():.6f}"
        f"+/- {cv_scores.std():.6f}"
    )

    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    predictions = predict_survival(
        model=pipeline,
        data=test,
    )

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions["Survived"],
    })

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Submission saved to: {SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")


if __name__ == "__main__":
    main()
