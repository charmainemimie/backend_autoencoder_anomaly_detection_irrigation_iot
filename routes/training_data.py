# routes/training_data.py
from flask import Blueprint, jsonify
from app import kiambu_data, kano_data

training_bp = Blueprint('training_data', __name__)

@training_bp.route('/api/training_data/<region>', methods=['GET'])
def get_training_data(region):
    if region == 'kiambu':
        data = kiambu_data.tail(100)
    elif region == 'kano':
        data = kano_data.tail(100)
    else:
        return jsonify({'error': 'Invalid region'}), 400
    
    result = data.to_dict('records')
    for record in result:
        record['timestamp'] = record['timestamp'].isoformat()
    
    return jsonify({
        'region': region,
        'samples': result,
        'total_training_samples': len(kiambu_data if region == 'kiambu' else kano_data)
    })
