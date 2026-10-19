import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")

st.write(
    """
    This application predicts whether a telecom customer is likely to churn
    by connecting to a live FastAPI Backend REST Service.
    """
)

# API Configuration
API_URL = "http://127.0.0.1:8000/predict"

st.header("Select Machine Learning Model")

model_name = st.selectbox(
    "Choose Model",
    (
        "XGBoost",
        "Random Forest",
        "Logistic Regression"
    )
)

st.header("Customer Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.subheader("Contract & Billing")

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=800.0)

if st.button("Predict"):
    payload = {
        "model_name": model_name,
        "gender": gender,
        "senior": senior,
        "partner": partner,
        "dependents": dependents,
        "phone": phone,
        "multiple": multiple,
        "internet": internet,
        "security": security,
        "backup": backup,
        "protection": protection,
        "support": support,
        "tv": tv,
        "movies": movies,
        "contract": contract,
        "paperless": paperless,
        "payment": payment,
        "tenure": tenure,
        "monthly": monthly,
        "total": total
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            
            st.subheader("Prediction Result")
            st.info(f"Model Used: **{result['model_used']}**")
            
            if result["prediction"] == 1:
                st.error(f"⚠️ **Customer is likely to Churn.** (Risk Probability: {result['churn_probability']}%)")
            else:
                st.success(f"✅ **Customer is NOT likely to Churn.** (Risk Probability: {result['churn_probability']}%)")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to FastAPI backend: {e}")