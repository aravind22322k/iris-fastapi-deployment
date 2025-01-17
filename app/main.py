from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the model
model = joblib.load("model/model.pkl")

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
