# tests/test_model.py - a simple sanity check for our model

import joblib
import pandas as pd
import os

def test_model_file_exists():
    """Check that our trained model file exists"""
    assert os.path.exists("house_price_model.pkl"), "Model file is missing!"

def test_model_can_predict():
    """Check that the model can actually make a prediction"""
    model = joblib.load("house_price_model.pkl")
    
    # Load column structure
    data = pd.read_csv("House_Price_Prediction_Dataset.csv")
    data = pd.get_dummies(data, columns=["Location", "Condition", "Garage"])
    model_columns = data.drop(columns=["Id", "Price"]).columns
    
    # Fake house data
    sample = pd.DataFrame([{
        "Area": 2000, "Bedrooms": 3, "Bathrooms": 2, "Floors": 1,
        "YearBuilt": 2005, "Location": "Downtown", "Condition": "Good", "Garage": "Yes"
    }])
    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(sample)
    
    # Just check we got ONE number back, and it's positive
    assert len(prediction) == 1, "Should predict exactly one price"
    assert prediction[0] > 0, "Price should be a positive number"