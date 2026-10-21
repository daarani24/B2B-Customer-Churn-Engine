from fastapi import FastAPI, HTTPException, Security, UploadFile, File, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import io
import os
from dotenv import load_dotenv
from backend.database import init_db, log_prediction

load_dotenv()

app = FastAPI(
    title="Customer Churn Prediction B2B API",
    description="Production REST API with Authentication, Batch Predictions, and Database Logging",
    version="2.0.0"
)

@app.on_event("startup")
def startup_event():
    init_db()

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "default_fallback_key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_SECRET_KEY:
        return api_key
    raise HTTPException(
        status_code=401,
        detail="Unauthorized: Invalid or Missing X-API-Key Header"
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

try:
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    model_columns = joblib.load(os.path.join(MODELS_DIR, "model_columns.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("All ML models loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

class CustomerData(BaseModel):
    model_name: str = Field(default="XGBoost")
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

def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    input_df = pd.DataFrame(0, index=df_raw.index, columns=model_columns)
    
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df_raw.columns:
            input_df[col] = df_raw[col]
            
    if "gender" in df_raw.columns:
        input_df.loc[df_raw["gender"] == "Male", "gender_Male"] = 1
    if "partner" in df_raw.columns:
        input_df.loc[df_raw["partner"] == "Yes", "Partner_Yes"] = 1
    if "dependents" in df_raw.columns:
        input_df.loc[df_raw["dependents"] == "Yes", "Dependents_Yes"] = 1
    if "phone" in df_raw.columns:
        input_df.loc[df_raw["phone"] == "Yes", "PhoneService_Yes"] = 1

    if "multiple" in df_raw.columns:
        input_df.loc[df_raw["multiple"] == "Yes", "MultipleLines_Yes"] = 1
        input_df.loc[df_raw["multiple"] == "No phone service", "MultipleLines_No phone service"] = 1

    if "internet" in df_raw.columns:
        input_df.loc[df_raw["internet"] == "Fiber optic", "InternetService_Fiber optic"] = 1
        input_df.loc[df_raw["internet"] == "No", "InternetService_No"] = 1

    if "security" in df_raw.columns:
        input_df.loc[df_raw["security"] == "Yes", "OnlineSecurity_Yes"] = 1
        input_df.loc[df_raw["security"] == "No internet service", "OnlineSecurity_No internet service"] = 1

    if "contract" in df_raw.columns:
        input_df.loc[df_raw["contract"] == "One year", "Contract_One year"] = 1
        input_df.loc[df_raw["contract"] == "Two year", "Contract_Two year"] = 1

    if "payment" in df_raw.columns:
        input_df.loc[df_raw["payment"] == "Electronic check", "PaymentMethod_Electronic check"] = 1
        input_df.loc[df_raw["payment"] == "Credit card (automatic)", "PaymentMethod_Credit card (automatic)"] = 1
        input_df.loc[df_raw["payment"] == "Mailed check", "PaymentMethod_Mailed check"] = 1

    return input_df

@app.get("/")
def home():
    return {"status": "online", "service": "B2B Churn Prediction API"}

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_single(data: CustomerData):
    raw_df = pd.DataFrame([data.dict()])
    processed_df = preprocess_input(raw_df)

    if data.model_name == "Logistic Regression":
        scaled = scaler.transform(processed_df)
        pred = int(lr_model.predict(scaled)[0])
        prob = float(lr_model.predict_proba(scaled)[0][1])
    elif data.model_name == "Random Forest":
        pred = int(rf_model.predict(processed_df)[0])
        prob = float(rf_model.predict_proba(processed_df)[0][1])
    else:
        pred = int(xgb_model.predict(processed_df)[0])
        prob = float(xgb_model.predict_proba(processed_df)[0][1])

    log_prediction(data.model_name, "single", pred, round(prob * 100, 2))

    return {
        "model_used": data.model_name,
        "prediction": pred,
        "churn_status": "Likely to Churn" if pred == 1 else "Not Likely to Churn",
        "churn_probability": round(prob * 100, 2)
    }

@app.post("/predict-batch", dependencies=[Depends(verify_api_key)])
async def predict_batch(file: UploadFile = File(...), model_name: str = "XGBoost"):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    raw_df = pd.read_csv(io.BytesIO(contents))
    processed_df = preprocess_input(raw_df)

    if model_name == "Logistic Regression":
        scaled = scaler.transform(processed_df)
        probs = lr_model.predict_proba(scaled)[:, 1]
    elif model_name == "Random Forest":
        probs = rf_model.predict_proba(processed_df)[:, 1]
    else:
        probs = xgb_model.predict_proba(processed_df)[:, 1]

    raw_df["churn_probability_%"] = (probs * 100).round(2)
    raw_df["churn_prediction"] = (probs > 0.5).astype(int)
    raw_df["churn_status"] = raw_df["churn_prediction"].map({1: "Likely to Churn", 0: "Not Likely to Churn"})

    log_prediction(model_name, "batch", int(raw_df["churn_prediction"].sum()), float(probs.mean() * 100))

    return raw_df.to_dict(orient="records")