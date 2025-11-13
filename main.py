from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Initialize the App
app = FastAPI(title="Fraud Detection API", description="MLOps Project")

# 2. Load the Model
# We use try/except to catch errors if the file is missing
try:
    model = joblib.load("fraud_detection_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# 3. Define the Data Structure (Validation)
# This ensures the user sends Numbers, not text like "abc"
class TransactionInput(BaseModel):
    amount: float
    old_balance: float
    new_balance: float

# 4. The Prediction Endpoint
@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Prepare data for the model (it expects a list of lists)
    # Corresponds to: [amount, old_balance, new_balance]
    features = np.array([[
        transaction.amount, 
        transaction.old_balance, 
        transaction.new_balance
    ]])
    
    # Make prediction
    prediction = model.predict(features) # Returns [0] or [1]
    probability = model.predict_proba(features) # Returns [[0.95, 0.05]]
    
    # Return result
    is_fraud = int(prediction[0]) # Convert numpy int to python int
    confidence = float(np.max(probability)) * 100
    
    return {
        "is_fraud": bool(is_fraud),
        "confidence_score": f"{confidence:.2f}%",
        "message": "🚨 FRAUD DETECTED" if is_fraud else "✅ Transaction Safe"
    }

# 5. Root Endpoint (Just to check if it works)
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}