# routes/model_info.py
from flask import Blueprint, jsonify
from app import models, features, kiambu_data, kano_data

model_info_bp = Blueprint('model_info', __name__)

@model_info_bp.route('/api/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'Isolation Forest',
        'features': features,
        'regions': list(models.keys()),
        'training_samples': {
            'kiambu': len(kiambu_data),
            'kano': len(kano_data)
        },
        'contamination_rate': 0.01,
        'detection_threshold': 0.6,
        'deployment': 'Edge (Gateway Device)',
        'inference_time': '< 50ms',
        'model_size': '< 1MB'
    })
