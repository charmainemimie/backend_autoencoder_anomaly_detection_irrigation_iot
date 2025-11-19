# routes/anomaly.py
from flask import Blueprint, request, jsonify
import numpy as np
from app import models, features

anomaly_bp = Blueprint('anomaly', __name__)

@anomaly_bp.route('/api/detect_anomaly', methods=['POST'])
def detect_anomaly():
    data = request.json
    region = data.get('region', 'kiambu')
    
    if region not in models:
        return jsonify({'error': 'Invalid region'}), 400
    
    X = np.array([
        data['moisture'],
        data['temperature'],
        data['nitrogen'],
        data['phosphorus'],
        data['potassium']
    ]).reshape(1, -1)
    
    model = models[region]
    prediction = model.predict(X)[0]
    raw_score = model.decision_function(X)[0]
    normalized_score = max(0, min(1, 0.5 - raw_score))
    
    return jsonify({
        'is_anomaly': prediction == -1 or normalized_score > 0.6,
        'anomaly_score': float(normalized_score),
        'prediction': int(prediction)
    })
