from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready REST API for predicting telecom customer churn",
    version="1.0.0"
)

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# --- Load Models & Transformers on Startup ---
try:
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    model_columns = joblib.load(os.path.join(MODELS_DIR, "model_columns.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("All ML models and transformers loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")

# --- Input Request Schema ---
class CustomerData(BaseModel):
    model_name: str = Field(default="XGBoost", description="Model choice: 'Logistic Regression', 'Random Forest', or 'XGBoost'")
    gender: str = "Female"
    senior: int = 0
    partner: str = "No"
    dependents: str = "No"
    phone: str = "Yes"
    multiple: str = "No"
    internet: str = "Fiber optic"
    security: str = "No"
    backup: str = "No"
    protection: str = "No"
    support: str = "No"
    tv: str = "No"
    movies: str = "No"
    contract: str = "Month-to-month"
    paperless: str = "Yes"
    payment: str = "Electronic check"
    tenure: int = 12
    monthly: float = 70.0
    total: float = 800.0

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Customer Churn Prediction API is up and running!"
    }

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # 1. Initialize input dataframe with zeros aligned to model_columns
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # 2. Assign Numerical Features
        input_df["SeniorCitizen"] = data.senior
        input_df["tenure"] = data.tenure
        input_df["MonthlyCharges"] = data.monthly
        input_df["TotalCharges"] = data.total

        # 3. Categorical Transformations (One-Hot Encoding Alignment)
        if data.gender == "Male":
            input_df["gender_Male"] = 1
        if data.partner == "Yes":
            input_df["Partner_Yes"] = 1
        if data.dependents == "Yes":
            input_df["Dependents_Yes"] = 1
        if data.phone == "Yes":
            input_df["PhoneService_Yes"] = 1

        if data.multiple == "Yes":
            input_df["MultipleLines_Yes"] = 1
        elif data.multiple == "No phone service":
            input_df["MultipleLines_No phone service"] = 1

        if data.internet == "Fiber optic":
            input_df["InternetService_Fiber optic"] = 1
        elif data.internet == "No":
            input_df["InternetService_No"] = 1

        if data.security == "Yes":
            input_df["OnlineSecurity_Yes"] = 1
        elif data.security == "No internet service":
            input_df["OnlineSecurity_No internet service"] = 1

        if data.backup == "Yes":
            input_df["OnlineBackup_Yes"] = 1
        elif data.backup == "No internet service":
            input_df["OnlineBackup_No internet service"] = 1

        if data.protection == "Yes":
            input_df["DeviceProtection_Yes"] = 1
        elif data.protection == "No internet service":
            input_df["DeviceProtection_No internet service"] = 1

        if data.support == "Yes":
            input_df["TechSupport_Yes"] = 1
        elif data.support == "No internet service":
            input_df["TechSupport_No internet service"] = 1

        if data.tv == "Yes":
            input_df["StreamingTV_Yes"] = 1
        elif data.tv == "No internet service":
            input_df["StreamingTV_No internet service"] = 1

        if data.movies == "Yes":
            input_df["StreamingMovies_Yes"] = 1
        elif data.movies == "No internet service":
            input_df["StreamingMovies_No internet service"] = 1

        if data.contract == "One year":
            input_df["Contract_One year"] = 1
        elif data.contract == "Two year":
            input_df["Contract_Two year"] = 1

        if data.paperless == "Yes":
            input_df["PaperlessBilling_Yes"] = 1

        if data.payment == "Credit card (automatic)":
            input_df["PaymentMethod_Credit card (automatic)"] = 1
        elif data.payment == "Electronic check":
            input_df["PaymentMethod_Electronic check"] = 1
        elif data.payment == "Mailed check":
            input_df["PaymentMethod_Mailed check"] = 1

        # 4. Model Selection & Prediction
        if data.model_name == "Logistic Regression":
            scaled_input = scaler.transform(input_df)
            prediction = int(lr_model.predict(scaled_input)[0])
            probability = float(lr_model.predict_proba(scaled_input)[0][1])
        elif data.model_name == "Random Forest":
            prediction = int(rf_model.predict(input_df)[0])
            probability = float(rf_model.predict_proba(input_df)[0][1])
        elif data.model_name == "XGBoost":
            prediction = int(xgb_model.predict(input_df)[0])
            probability = float(xgb_model.predict_proba(input_df)[0][1])
        else:
            raise HTTPException(status_code=400, detail=f"Invalid model name: {data.model_name}")

        # 5. Format Output Response
        return {
            "model_used": data.model_name,
            "prediction": prediction,
            "churn_status": "Likely to Churn" if prediction == 1 else "Not Likely to Churn",
            "churn_probability": round(probability * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))