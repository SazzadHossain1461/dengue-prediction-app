# FILE 8: crud.py
# Save in project root

from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime, timedelta
import models, schemas

def create_patient(db: Session, patient: schemas.PatientCreate):
    db_patient = models.Patient(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

def get_patients(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Patient).offset(skip).limit(limit).all()

def get_patient(db: Session, patient_id: int):
    return db.query(models.Patient).filter(models.Patient.id == patient_id).first()

def delete_patient(db: Session, patient_id: int):
    db_patient = get_patient(db, patient_id)
    if db_patient:
        db.delete(db_patient)
        db.commit()
        return True
    return False

def create_prediction(db: Session, prediction: schemas.PredictionCreate, result: dict):
    db_prediction = models.Prediction(
        **prediction.dict(),
        prediction=result['prediction'],
        confidence=result['confidence']
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_predictions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Prediction).order_by(desc(models.Prediction.created_at)).offset(skip).limit(limit).all()

def get_prediction(db: Session, prediction_id: int):
    return db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()

def delete_prediction(db: Session, prediction_id: int):
    db_prediction = get_prediction(db, prediction_id)
    if db_prediction:
        db.delete(db_prediction)
        db.commit()
        return True
    return False

def get_analytics_summary(db: Session):
    total = db.query(func.count(models.Prediction.id)).scalar() or 0
    positive = db.query(func.count(models.Prediction.id)).filter(models.Prediction.prediction == 1).scalar() or 0
    negative = total - positive
    avg_confidence = db.query(func.avg(models.Prediction.confidence)).scalar() or 0
    
    return {
        'total_predictions': total,
        'positive_cases': positive,
        'negative_cases': negative,
        'positive_percentage': (positive / total * 100) if total > 0 else 0,
        'average_confidence': float(avg_confidence) if avg_confidence else 0
    }

def get_analytics_trends(db: Session, days: int = 30):
    start_date = datetime.utcnow() - timedelta(days=days)
    results = db.query(
        func.date(models.Prediction.created_at).label('date'),
        func.count(models.Prediction.id).label('total'),
        func.sum(models.Prediction.prediction).label('positive')
    ).filter(models.Prediction.created_at >= start_date).group_by(
        func.date(models.Prediction.created_at)
    ).all()
    
    return [
        {
            'date': str(row[0]),
            'total': row[1],
            'positive': row[2] or 0,
            'negative': row[1] - (row[2] or 0)
        }
        for row in results
    ]
