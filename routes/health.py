# routes/health.py
from flask import Blueprint, jsonify
from app import models
from datetime import datetime

health_bp = Blueprint('health', __name__)

@health_bp.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': list(models.keys())
    })
