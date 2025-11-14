# FILE 5: models.py
# Save in project root

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    gender = Column(String(10), nullable=False)
    age = Column(Integer, nullable=False)
    area = Column(String(100), nullable=False)
    district = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    
    gender = Column(String(10), nullable=False)
    age = Column(Integer, nullable=False)
    ns1 = Column(Integer, nullable=False)
    igg = Column(Integer, nullable=False)
    igm = Column(Integer, nullable=False)
    area = Column(String(100), nullable=False)
    area_type = Column(String(20), nullable=False)
    house_type = Column(String(20), nullable=False)
    district = Column(String(100), nullable=False)
    
    prediction = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="predictions")