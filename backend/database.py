import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_used = Column(String)
    input_type = Column(String)
    prediction = Column(Integer)
    probability = Column(Float)

def init_db():
    if engine:
        Base.metadata.create_all(bind=engine)

def log_prediction(model_used: str, input_type: str, prediction: int, probability: float):
    if not SessionLocal:
        return None
    db = SessionLocal()
    try:
        log_entry = PredictionLog(
            model_used=model_used,
            input_type=input_type,
            prediction=prediction,
            probability=probability
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        return log_entry
    except Exception as e:
        db.rollback()
        print(f"Error logging to database: {e}")
        return None
    finally:
        db.close()