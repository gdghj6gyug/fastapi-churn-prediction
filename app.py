from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
app = FastAPI()
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is Running!"}
model = joblib.load("final_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

def preprocess_new_data(data):
    df = pd.DataFrame([data])

    # Compute loyalty_score automatically
    df["loyalty_score"] = df["totalcharges"] / (df["tenure"] + 0.01)
    
    # Scale the necessary features
    df[["tenure", "monthlycharges", "totalcharges","loyalty_score"]] = scaler.transform(
        df[["tenure", "monthlycharges", "totalcharges","loyalty_score"]]
    )

    # Ensure the order of features matches training
    df = df[features]

    return df

@app.post("/predict")
def predict_churn(customer_data: dict):
    try:
        # Preprocess input data
        processed_data = preprocess_new_data(customer_data)

        # **DEBUG STEP: Print feature names**
        print("Train Columns:", features)
        print("Test Columns:", processed_data.columns.tolist())

        # Ensure correct feature order
        processed_data = processed_data[features]  # Fix feature mismatch

        # Make prediction
        probabilities = model.predict_proba(processed_data)
        final_prediction = (probabilities[:, 1] >= 0.45).astype(int)

        return {
            "churn_probability": float(probabilities[0, 1]),
            "prediction": int(final_prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}
