from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from iris_service.features import to_model_input, InvalidFeatureError
from iris_service.model import load_model, predict_species

app = FastAPI(title="Iris Species Predictor")
_model_bundle = load_model() # returns the joblib file that contains the model


class IrisMeasurements(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(measurements: IrisMeasurements):
    try:
        vector = to_model_input(measurements.model_dump())
        # to_model_input calls validate_measurements, which should validate the input dictionary and return an ordered list of floats
    except InvalidFeatureError as e:
        raise HTTPException(status_code=422, detail=str(e)) # if validate_measurements raises an invalidfeatureerror give it a 422 code and details
    species = predict_species(_model_bundle, vector) # run the trained model
    return {"species": species}