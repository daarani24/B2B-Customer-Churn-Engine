# 📊 B2B Customer Churn Engine

[![CI Pipeline](https://github.com/daarani24/B2B-Customer-Churn-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/daarani24/B2B-Customer-Churn-Engine/actions)

An end-to-end, containerized machine learning platform built to predict enterprise customer churn and protect Monthly Recurring Revenue (MRR). The platform features a trained XGBoost pipeline, a secure FastAPI REST backend with custom API key authentication, automated cloud logging via SQLAlchemy to Neon PostgreSQL, and an interactive Streamlit UI.

---

## 🏗️ System Architecture

```text
┌─────────────────────────┐
│     USER / BROWSER      │
└────────────┬────────────┘
             │ 1. Account attributes input
             ▼
┌─────────────────────────┐
│  STREAMLIT UI (:8501)   │
│    dashboard/app.py     │
└────────────┬────────────┘
             │ 2. HTTP POST /predict (Header: X-API-Key)
             ▼
┌────────────────────────────────────────────────────────┐
│                 FASTAPI BACKEND (:8000)                │
│                     backend/main.py                    │
│                                                        │
│  - Authenticate X-API-Key                              │
│  - Validate schema via Pydantic                        │
│  - Apply scaler.pkl & model_columns.pkl                │
│  - Execute xgboost_model.pkl inference                 │
└────────────┬─────────────────────────────┬─────────────┘
             │ 3. log_prediction()         │ 4. Return JSON
             ▼                             ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│  NEON POSTGRESQL (Cloud)│   │  STREAMLIT RESULT UI     │
│ table: prediction_logs  │   │  Color-coded outcome     │
└─────────────────────────┘   └──────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Machine Learning** | Python 3.10+, Scikit-Learn, XGBoost, Pandas, NumPy, Joblib |
| **Backend API** | FastAPI, Uvicorn, Pydantic, `python-dotenv` |
| **Database & ORM** | Neon PostgreSQL (Cloud), SQLAlchemy ORM |
| **Security** | Custom Header Auth (`X-API-Key`), Git-ignored `.env` secret management |
| **Frontend UI** | Streamlit, Requests |
| **DevOps & Containers** | Docker, Docker Compose, Git |

---

## 🧠 Machine Learning Pipeline

### Dataset & Preprocessing
* **Source:** Kaggle Telco Customer Churn dataset (**7,043 customer records**, 21 features).
* **Missing Values:** Blank string charges converted to numerical float and imputed using median values.
* **Feature Scaling:** `StandardScaler` applied to continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`).
* **Categorical Encoding:** `OneHotEncoder` for nominal attributes (`InternetService`, `PaymentMethod`) and ordinal mapping for binary/contract types (`Contract`, `PaperlessBilling`).
* **Data Splitting:** 80/20 Stratified Split (`stratify=y`) preserving the ~26.5% baseline positive churn class ratio (5,634 train / 1,409 test rows).

### Model Benchmarks & Metrics

| Model | Status | ROC-AUC | Recall (Positive Class) |
|---|---|---|---|
| Logistic Regression | Baseline | ~0.80 | ~0.72 |
| Random Forest | Benchmark | ~0.82 | ~0.75 |
| **XGBoost Classifier** | **Production Selected** | **~0.84 - 0.85** | **~0.78 - 0.80** |

* **ROC-AUC (Primary):** Selected to measure continuous probability class separation independent of arbitrary decision thresholds.
* **Recall (Secondary):** Prioritized over accuracy to minimize **false negatives** (unflagged churning accounts, which cause permanent MRR loss).

---

## 📡 API Reference

### Headers
```http
Content-Type: application/json
X-API-Key: <YOUR_API_SECRET_KEY>
```

### `POST /predict`
Submit customer attributes to evaluate churn probability.

**Sample Request Payload:**
```json
{
  "model_name": "XGBoost",
  "gender": "Female",
  "senior": 0,
  "partner": "Yes",
  "dependents": "No",
  "phone": "Yes",
  "multiple": "No",
  "internet": "Fiber optic",
  "security": "No",
  "backup": "Yes",
  "protection": "No",
  "support": "No",
  "tv": "Yes",
  "movies": "No",
  "contract": "Month-to-month",
  "paperless": "Yes",
  "payment": "Electronic check",
  "tenure": 12,
  "monthly": 70.35,
  "total": 844.20
}
```

**Sample Response (`200 OK`):**
```json
{
  "model_used": "XGBoost",
  "prediction": 0,
  "churn_status": "Not Likely to Churn",
  "churn_probability": 11.53
}
```

---

## 🚀 Quickstart Guide

### 1. Environment Configuration
Clone the repository and set up environment variables:

```bash
git clone https://github.com/daarani24/B2B-customer-churn-engine.git
cd B2B-customer-churn-engine
cp .env.example .env
```

Configure your local `.env` file:
```env
API_SECRET_KEY=your_secret_api_key_here
DATABASE_URL=postgresql://user:password@ep-example.neon.tech/neondb?sslmode=require
BACKEND_URL=http://backend:8000
```

### 2. Multi-Container Orchestration (Docker Compose)
Launch the entire backend, frontend, and cloud connection stack in containerized isolation:

```bash
docker compose up --build
```

Access the interfaces:
* **Streamlit Dashboard:** `http://localhost:8501`
* **FastAPI Backend / OpenAPI Docs:** `http://localhost:8000/docs`

To stop services:
```bash
docker compose down
```

---

## 🗄️ Database Schema (`prediction_logs`)

All inference outputs automatically record an audit trail to Neon PostgreSQL via SQLAlchemy ORM.

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-incrementing primary key |
| `timestamp` | DateTime | UTC prediction timestamp |
| `model_used` | String | Machine learning model name |
| `prediction` | Integer | Binary class (`0` = Retained, `1` = Churn) |
| `probability` | Float | Calculated churn probability percentage |

---

## 📂 Project Structure

```text
B2B-customer-churn-engine/
│
├── backend/
│   ├── main.py              # FastAPI app, authentication, Pydantic schemas
│   └── database.py          # SQLAlchemy ORM, Neon DB connection, logging functions
│
├── dashboard/
│   └── app.py               # Streamlit UI with dynamic backend routing
│
├── data/
│   ├── raw/                 # Raw Telco CSV dataset
│   └── processed/           # Feature engineered arrays
│
├── models/
│   ├── xgboost_model.pkl    # Production model binary
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl           # Fitted StandardScaler artifact
│   └── model_columns.pkl    # One-hot encoding column layout
│
├── notebooks/
│   └── Customer_Churn_Analysis.ipynb # EDA, preprocessing, model benchmarking
│
├── Dockerfile               # Service container definition
├── docker-compose.yml       # Multi-service stack specification
├── .env.example             # Safe environment variable template
├── .gitignore               # Configured secret & binary exclusion
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 📄 License & Author

* **License:** MIT License
* **Author:** Daarani (B.Tech, Artificial Intelligence & Data Science)