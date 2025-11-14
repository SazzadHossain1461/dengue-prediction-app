# main.py - Updated for Hybrid Model Integration
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import models, schemas, crud
from database import get_db, init_db
from ml_utils import model_manager
from config import get_settings
import uvicorn

settings = get_settings()

# Create app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Dengue Prediction System with Hybrid AI Model"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
@app.on_event("startup")
def startup():
    init_db()
    print(f"✓ Backend running on http://{settings.host}:{settings.port}")
    print(f"✓ API Docs: http://localhost:{settings.port}/docs")
    print(f"✓ Model Status: {model_manager.get_model_info()['status']}")

# ============ HEALTH CHECK ============
@app.get("/api/health")
async def health_check():
    model_info = model_manager.get_model_info()
    return {
        "status": "healthy",
        "database": "connected",
        "model": model_info["status"],
        "model_type": model_info["model_type"],
        "version": settings.api_version
    }

# ============ PATIENTS ============
@app.post("/api/patients", response_model=schemas.Patient)
async def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    return crud.create_patient(db, patient)

@app.get("/api/patients", response_model=list[schemas.Patient])
async def get_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_patients(db, skip, limit)

@app.get("/api/patients/{patient_id}", response_model=schemas.Patient)
async def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    if not crud.delete_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"message": "Patient deleted successfully"}

# ============ PREDICTIONS ============
@app.post("/api/predictions", response_model=schemas.PredictionResponse)
async def create_prediction(pred: schemas.PredictionCreate, db: Session = Depends(get_db)):
    try:
        # Get prediction from hybrid model
        prediction, confidence = model_manager.predict(
            gender=pred.gender,
            age=pred.age,
            ns1=pred.ns1,
            igg=pred.igg, 
            igm=pred.igm,
            area_type=pred.area_type,
            house_type=pred.house_type,
            area=pred.area,
            district=pred.district
        )
        
        result = {'prediction': prediction, 'confidence': float(confidence)}
        db_pred = crud.create_prediction(db, pred, result)
        
        return {
            "id": db_pred.id,
            "patient_id": db_pred.patient_id,
            "prediction": prediction,
            "confidence": float(confidence),
            "risk_level": "High" if prediction == 1 else "Low",
            "created_at": db_pred.created_at,
            "message": "High dengue risk detected" if prediction == 1 else "Low dengue risk",
            "model_type": model_manager.get_model_info()["model_type"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/api/predictions", response_model=list[schemas.Prediction])
async def get_predictions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_predictions(db, skip, limit)

@app.get("/api/predictions/{prediction_id}", response_model=schemas.Prediction)
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    pred = crud.get_prediction(db, prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return pred

@app.delete("/api/predictions/{prediction_id}")
async def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    if not crud.delete_prediction(db, prediction_id):
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"message": "Prediction deleted successfully"}

# ============ ANALYTICS ============
@app.get("/api/analytics/summary", response_model=schemas.AnalyticsSummary)
async def get_summary(db: Session = Depends(get_db)):
    return crud.get_analytics_summary(db)

@app.get("/api/analytics/trends")
async def get_trends(days: int = 30, db: Session = Depends(get_db)):
    return crud.get_analytics_trends(db, days)

# ============ MODEL ENDPOINTS ============
@app.get("/api/model/info")
async def get_model_info():
    """Get information about the loaded ML model"""
    return model_manager.get_model_info()

@app.post("/api/model/retrain")
async def retrain_model():
    """Retrain the model with current dataset"""
    try:
        # This would trigger retraining - for now just return info
        return {
            "message": "Model retraining endpoint",
            "status": "Use train_model.py to retrain the model",
            "dataset": "dataset.csv"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# ============ DATASET INFO ============
@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get information about the training dataset"""
    try:
        import pandas as pd
        df = pd.read_csv('dataset.csv')
        return {
            "dataset_size": len(df),
            "positive_cases": int(df['Outcome'].sum()),
            "negative_cases": int(len(df) - df['Outcome'].sum()),
            "features": list(df.columns),
            "positive_rate": f"{(df['Outcome'].mean() * 100):.1f}%"
        }
    except Exception as e:
        return {"error": f"Could not load dataset: {str(e)}"}

# ============ ROOT ============
@app.get("/")
async def root():
    model_info = model_manager.get_model_info()
    return {
        "message": "Dengue Prediction API with Hybrid AI",
        "version": settings.api_version,
        "docs": f"http://localhost:{settings.port}/docs",
        "model": model_info["model_type"],
        "status": model_info["status"],
        "dataset": "Integrated real dengue dataset"
    }

# ============ RUN ============
if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)