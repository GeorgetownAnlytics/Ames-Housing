import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from flask import Flask, request, jsonify,render_template
from src.AmesFeatureEngineer import AmesFeatureEngineer as AFE
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
engineer = AFE()

@app.route('/')
def index():
    return render_template('index.html', google_maps_api_key=os.getenv('GOOGLE_MAPS_API_KEY'))

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
