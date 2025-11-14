# FILE 6: schemas.py
# Save in project root

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class PatientBase(BaseModel):
    name: Optional[str] = None
    gender: str
    age: int = Field(..., ge=0, le=120)
    area: str
    district: str
    phone: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class Patient(PatientBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class PredictionBase(BaseModel):
    gender: str
    age: int = Field(..., ge=0, le=120)
    ns1: int = Field(..., ge=0, le=1)
    igg: int = Field(..., ge=0, le=1)
    igm: int = Field(..., ge=0, le=1)
    area: str
    area_type: str
    house_type: str
    district: str
    patient_id: Optional[int] = None

class PredictionCreate(PredictionBase):
    pass

class Prediction(PredictionBase):
    id: int
    prediction: int
    confidence: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    id: int
    prediction: int
    confidence: float
    risk_level: str
    created_at: datetime

class AnalyticsSummary(BaseModel):
    total_predictions: int
    positive_cases: int
    negative_cases: int
    positive_percentage: float
    average_confidence: float