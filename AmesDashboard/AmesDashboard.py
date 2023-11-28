import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from flask import Flask, request, jsonify
from src.AmesFeatureEngineer import AmesFeatureEngineer as AFE

app = Flask(__name__)
engineer = AmesFeatureEngineer()

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.json
    input_df = pd.DataFrame([data])
    
    # Apply feature engineering
    transformed_data = engineer.transform(input_df)
    
    # Load model and make predictions (assuming model is loaded)
    prediction = model.predict(transformed_data)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
