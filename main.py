# Put the code for your API here.

import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference
import pandas as pd

# List of categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class UserProfile(BaseModel):
    age: int = Field(examples=[40])
    workclass: str = Field(examples=["Privte"], description="Type of work class")
    fnlgt: int = Field(
        examples=[11111], description="Final weight (a numerical attribute)"
    )
    education: str = Field(examples=["Bachelors"])
    education_num: int = Field(alias="education-num", examples=[9])
    marital_status: str = Field(alias="marital-status", examples=["Never-married"])
    occupation: str = Field(examples=["Prof-specialty"])
    relationship: str = Field(examples=["Not-in-family"])
    race: str = Field(examples=["White"])
    sex: str = Field(examples=["Male"])
    capital_gain: int = Field(alias="capital-gain", examples=[1111])
    capital_loss: int = Field(alias="capital-loss", examples=[0])
    hours_per_week: int = Field(alias="hours-per-week", examples=[60])
    native_country: str = Field(alias="native-country", examples=["United-States"])


# Load model and preprocessing objects
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

# Create FastAPI app
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/inference")
def run_inference(data: UserProfile):
    input_df = pd.DataFrame([data.dict(by_alias=True)])

    # Process the input data
    X_processed, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run inference
    predictions = inference(model, X_processed)

    # Convert predictions to list for JSON response
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    pass
