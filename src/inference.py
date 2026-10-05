import pandas as pd


def predict_survival(model, data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict Titanic survival classes.

    Parameters
    ----------
    model
        Fitted sklearn-compatible classification pipeline.

    data : pd.DataFrame
        Input passenger features.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column, ``Survived``,
        containing binary predictions.
    """

    predictions = model.predict(data).astype(int)

    return pd.DataFrame({
        "Survived": predictions
    })
