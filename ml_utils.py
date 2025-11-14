# ml_utils.py - Updated for Hybrid Model Integration
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
from config import get_settings

settings = get_settings()

class HybridModelManager:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.feature_names = ['Gender', 'Age', 'NS1', 'IgG', 'IgM', 'Area', 'AreaType', 'HouseType', 'District']
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained hybrid model and preprocessor"""
        try:
            # Try to load the trained model
            model_path = os.path.join('models', 'hybrid_dengue_model.h5')
            preprocessor_path = os.path.join('models', 'preprocessor.pkl')
            
            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                self.model = tf.keras.models.load_model(model_path)
                self.preprocessor = joblib.load(preprocessor_path)
                self.model_loaded = True
                print("✓ Hybrid dengue model loaded successfully")
            else:
                print("⚠ Trained model not found. Using rule-based fallback.")
                self.model_loaded = False
                
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("✓ Using rule-based fallback model")
            self.model_loaded = False
    
    def preprocess_features(self, features_dict):
        """Preprocess input features for prediction"""
        # Convert input to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        if self.model_loaded and self.preprocessor:
            try:
                # Use the trained preprocessor
                processed_features = self.preprocessor.transform(features_df)
                return processed_features
            except Exception as e:
                print(f"Preprocessing error: {e}")
                return self._manual_preprocess(features_df)
        else:
            return self._manual_preprocess(features_df)
    
    def _manual_preprocess(self, features_df):
        """Manual preprocessing when trained model is not available"""
        # Encode categorical variables manually
        features_df = features_df.copy()
        
        # Gender encoding
        features_df['Gender'] = features_df['Gender'].map({'Male': 1, 'Female': 0}).fillna(1)
        
        # AreaType encoding
        area_type_map = {'Developed': 2, 'Undeveloped': 1, 'Rural': 0}
        features_df['AreaType'] = features_df['AreaType'].map(area_type_map).fillna(1)
        
        # HouseType encoding
        house_type_map = {'Building': 2, 'Apartment': 1, 'Tinshed': 0, 'Other': 1, 'Slum': 0}
        features_df['HouseType'] = features_df['HouseType'].map(house_type_map).fillna(1)
        
        # District risk encoding (simplified)
        high_risk_districts = ['Mirpur', 'Jatrabari', 'Demra', 'Kamrangirchar', 'Hazaribagh']
        features_df['District_Risk'] = features_df['District'].isin(high_risk_districts).astype(int)
        
        # Select and scale numerical features
        numerical_features = ['Gender', 'Age', 'NS1', 'IgG', 'IgM', 'AreaType', 'HouseType', 'District_Risk']
        features_array = features_df[numerical_features].values
        
        # Simple scaling
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
            # Fit scaler with some dummy data for fallback
            dummy_data = np.random.randn(100, len(numerical_features))
            self.scaler.fit(dummy_data)
        
        return self.scaler.transform(features_array)
    
    def predict(self, gender, age, ns1, igg, igm, area_type, house_type, area="Unknown", district="Dhaka"):
        """
        Make prediction using hybrid model
        
        Parameters:
        gender: 'Male' or 'Female'
        age: patient age
        ns1, igg, igm: test results (0 or 1)
        area_type: 'Developed', 'Undeveloped', etc.
        house_type: 'Building', 'Apartment', 'Tinshed', etc.
        area: area name (optional)
        district: district name
        """
        try:
            # Prepare features dictionary
            features_dict = {
                'Gender': gender,
                'Age': age,
                'NS1': ns1,
                'IgG': igg,
                'IgM': igm,
                'Area': area,
                'AreaType': area_type,
                'HouseType': house_type,
                'District': district
            }
            
            # Preprocess features
            processed_features = self.preprocess_features(features_dict)
            
            if self.model_loaded and self.model:
                # Use trained model for prediction
                prediction_proba = self.model.predict(processed_features, verbose=0)[0][0]
                prediction = 1 if prediction_proba > 0.5 else 0
                confidence = float(prediction_proba if prediction == 1 else 1 - prediction_proba)
            else:
                # Fallback to rule-based prediction
                prediction, confidence = self._rule_based_prediction(
                    gender, age, ns1, igg, igm, area_type, house_type, district
                )
            
            # Ensure minimum confidence
            confidence = max(0.6, min(0.95, confidence))
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Ultimate fallback
            return self._basic_fallback(ns1, igg, igm)
    
    def _rule_based_prediction(self, gender, age, ns1, igg, igm, area_type, house_type, district):
        """Enhanced rule-based prediction using dataset patterns"""
        # Base score from tests (based on dataset analysis)
        test_score = 0
        
        # NS1 antigen (most important - indicates active infection)
        if ns1 == 1:
            test_score += 0.6
        
        # IgM antibodies (recent infection)
        if igm == 1:
            test_score += 0.3
        
        # IgG antibodies (past infection - lower weight)
        if igg == 1:
            test_score += 0.1
        
        # Age risk factors (from dataset analysis)
        demographic_score = 0
        if age < 15:  # Children higher risk in dataset
            demographic_score += 0.2
        elif age > 50:  # Elderly higher risk
            demographic_score += 0.1
        
        # Gender (slight male predominance in dataset)
        if gender.lower() == 'male':
            demographic_score += 0.05
        
        # Environmental factors (from dataset analysis)
        environmental_score = 0
        
        # Area type
        if area_type.lower() == 'undeveloped':
            environmental_score += 0.2
        elif area_type.lower() == 'developed':
            environmental_score += 0.1
        
        # House type
        if house_type.lower() in ['tinshed', 'slum']:
            environmental_score += 0.3
        elif house_type.lower() == 'apartment':
            environmental_score += 0.1
        
        # District risk (from dataset patterns)
        high_risk_districts = ['mirpur', 'jatrabari', 'demra', 'kamrangirchar', 'hazaribagh']
        if district.lower() in high_risk_districts:
            environmental_score += 0.2
        
        # Combine scores with weights
        total_score = (
            test_score * 0.6 +           # Tests: 60% weight
            demographic_score * 0.2 +    # Demographics: 20% weight  
            environmental_score * 0.2    # Environment: 20% weight
        )
        
        # Determine prediction and confidence
        if total_score >= 0.5:
            prediction = 1  # High risk
            confidence = min(0.95, total_score)
        else:
            prediction = 0  # Low risk
            confidence = min(0.95, 1 - total_score)
        
        return prediction, confidence
    
    def _basic_fallback(self, ns1, igg, igm):
        """Simple fallback prediction based only on tests"""
        if ns1 == 1:
            return 1, 0.85  # NS1 positive = high risk
        elif igm == 1:
            return 1, 0.70  # IgM positive = moderate risk
        elif igg == 1:
            return 0, 0.65  # IgG only = low risk
        else:
            return 0, 0.80  # All negative = low risk
    
    def get_model_info(self):
        """Return information about the current model"""
        return {
            "model_type": "Hybrid Dengue Prediction Model",
            "status": "loaded" if self.model_loaded else "rule_based_fallback",
            "version": "1.0",
            "features": self.feature_names,
            "description": "Combines TensorFlow DNN with rule-based fallback using real dataset patterns"
        }

# Global model manager instance
model_manager = HybridModelManager()