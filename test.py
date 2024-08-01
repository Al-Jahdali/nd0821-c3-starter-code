from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import (
    train_model,
    inference,
    compute_model_metrics,
)
import numpy as np
import joblib
import pytest

model = joblib.load("model/model.pkl")


def test_compute_model_metrics_values():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F-beta should be between 0 and 1"


def test_loading_data():

    data = pd.read_csv("./data/census.csv")
    assert isinstance(data, pd.DataFrame)


def test_load_model():
    # X_train, _, y_train, _ = split_data
    # model = train_model(X_train, y_train)

    model = joblib.load("model/model.pkl")

    assert isinstance(
        model, RandomForestClassifier
    ), "Model should be a RandomForestClassifier"
