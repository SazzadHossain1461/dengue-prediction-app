# train_model.py - Train your hybrid model with the dataset
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_hybrid_model():
    print("🚀 Training Hybrid Dengue Prediction Model...")
    
    # Load your dataset
    df = pd.read_csv('dataset.csv')
    print(f"Dataset loaded: {df.shape}")
    print(f"Outcome distribution:\n{df['Outcome'].value_counts()}")
    
    # Prepare features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Define feature types
    numerical_features = ['Age', 'NS1', 'IgG', 'IgM']
    categorical_features = ['Gender', 'Area', 'AreaType', 'HouseType', 'District']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess features
    print("Preprocessing features...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Build TensorFlow model
    print("Building model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_processed.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train_processed, y_train,
        validation_data=(X_test_processed, y_test),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
        ]
    )
    
    # Evaluate model
    print("Evaluating model...")
    y_pred_proba = model.predict(X_test_processed).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and preprocessor
    model.save('models/hybrid_dengue_model.h5')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    print("✅ Model saved successfully!")
    print("📁 Files created:")
    print("   - models/hybrid_dengue_model.h5")
    print("   - models/preprocessor.pkl")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300)
    print("   - models/training_history.png")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300)
    print("   - models/confusion_matrix.png")
    
    return model, preprocessor, accuracy, auc_score

if __name__ == "__main__":
    train_hybrid_model()