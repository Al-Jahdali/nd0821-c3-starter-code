import pytest
from fastapi.testclient import TestClient
from main import app  # Make sure to import your FastAPI app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_inference_valid_data():
    # Example valid data
    data = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 11111,
        "education": "Bachelors",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 1111,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }

    response = client.post("/inference", json=data)
    assert response.status_code == 200
    assert "predictions" in response.json()


def test_inference_missing_field():
    # Example data with a missing required field
    data = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 11111,
        # Missing 'education'
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 1111,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }

    response = client.post("/inference", json=data)
    assert response.status_code == 422  # Unprocessable Entity, due to validation error


def test_inference_valid_data_content():
    # Example valid data
    data = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 11111,
        "education": "Bachelors",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 1111,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }

    response = client.post("/inference", json=data)
    assert response.text == '{"predictions":[0]}'
