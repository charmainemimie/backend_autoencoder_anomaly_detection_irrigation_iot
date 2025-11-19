from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import time
from tensorflow import keras
import pickle
import os
import json
import config  # Import configuration

app = Flask(__name__)
CORS(app)

# Global state
current_readings = {
    'moisture': 45.0,
    'temperature': 25.0,
    'valve': 0,
    'valve_flow_rate': 0.0,  # L/min
    'timestamp': datetime.now().isoformat(),
    'anomaly_detected': False,
    'anomaly_score': 0.0
}

attack_active = False
attack_params = {'target': None, 'intensity': 0}

# User credentials
USERS = {
    'admin': {
        'password': 'admin123', 
        'name': 'John Kamau', 
        'farm': 'East Africa - Nairobi East Farm',
        'region': 'East Africa'
    }
}

# Model variables
autoencoder = None
scaler = None
anomaly_threshold = 0.05

def load_trained_model():
    """Load the trained autoencoder and scaler"""
    global autoencoder, scaler, anomaly_threshold
    
    try:
        if os.path.exists('autoencoder_model.h5') and os.path.exists('scaler.pkl'):
            print("\n" + "="*60)
            print("LOADING TRAINED MODEL")
            print("="*60)
            
            autoencoder = keras.models.load_model('autoencoder_model.h5')
            
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            print("✓ Autoencoder model loaded successfully")
            print("✓ Feature scaler loaded successfully")
            
            # Load and display metrics
            if os.path.exists('model_metrics.json'):
                with open('model_metrics.json', 'r') as f:
                    metrics = json.load(f)
                
                print("\n" + "="*60)
                print("MODEL PERFORMANCE METRICS")
                print("="*60)
                
                print(f"\nDataset Information:")
                for key, value in metrics.get('dataset_info', {}).items():
                    print(f"  {key}: {value}")
                
                print(f"\nFeature Statistics:")
                for key, value in metrics.get('feature_statistics', {}).items():
                    print(f"  {key}: {value:.2f}")
                
                print(f"\nModel Performance:")
                for key, value in metrics.get('model_performance', {}).items():
                    print(f"  {key}: {value:.6f}")
                
                print(f"\nAnomaly Detection:")
                for key, value in metrics.get('anomaly_detection', {}).items():
                    print(f"  {key}: {value}")
                
                # Set threshold from metrics
                anomaly_threshold = metrics.get('anomaly_detection', {}).get('threshold', 0.05)
                print(f"\n✓ Anomaly threshold set to: {anomaly_threshold:.6f}")
                print("="*60 + "\n")
        else:
            print("\n⚠ WARNING: Model files not found!")
            print("Please run train_model.py first to train the model.")
            print("The system will run without anomaly detection.\n")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("The system will run without anomaly detection.\n")
        autoencoder = None
        scaler = None

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if username in USERS and USERS[username]['password'] == password:
        return jsonify({
            'success': True,
            'user': {
                'username': username,
                'name': USERS[username]['name'],
                'farm': USERS[username]['farm'],
                'region': USERS[username]['region']
            }
        })
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/api/readings', methods=['GET'])
def get_readings():
    return jsonify(current_readings)

@app.route('/api/simulate-attack', methods=['POST'])
def simulate_attack():
    global attack_active, attack_params
    data = request.json
    attack_active = data.get('active', False)
    attack_params = {
        'target': data.get('target'),
        'intensity': data.get('intensity', 0)
    }
    
    if attack_active:
        print(f"\n🚨 ATTACK SIMULATION STARTED: {attack_params['target']} with intensity {attack_params['intensity']}")
    else:
        print(f"\n✓ Attack simulation stopped")
    
    return jsonify({'success': True})

def update_sensor_readings():
    """Background thread to simulate real-time sensor updates"""
    global current_readings, attack_active, attack_params
    
    # Use configuration values
    base_moisture = config.BASE_MOISTURE
    base_temperature = config.BASE_TEMPERATURE
    
    print("\n🌱 Starting real-time sensor monitoring...")
    print(f"📊 Base values: Moisture={base_moisture}%, Temperature={base_temperature}°C")
    print(f"📊 Variation: Moisture±{config.MOISTURE_STD}%, Temperature±{config.TEMPERATURE_STD}°C")
    print(f"📊 Threshold multiplier: {config.THRESHOLD_MULTIPLIER}x")
    print(f"📊 Updating readings every {config.UPDATE_INTERVAL} second(s)\n")
    
    while True:
        # Simulate realistic sensor variations using config values
        moisture = base_moisture + np.random.normal(0, config.MOISTURE_STD)
        temperature = base_temperature + np.random.normal(0, config.TEMPERATURE_STD)
        
        # Apply attack if active
        if attack_active:
            if attack_params['target'] == 'moisture':
                moisture += attack_params['intensity']
            elif attack_params['target'] == 'temperature':
                temperature += attack_params['intensity']
        
        # Ensure realistic bounds
        moisture = max(0, min(100, moisture))
        temperature = max(0, min(50, temperature))
        
        # Irrigation valve logic (using config values)
        valve = 1 if moisture < config.MIN_MOISTURE_THRESHOLD or temperature > config.MAX_TEMPERATURE_THRESHOLD else 0
        
        # Detect anomaly using autoencoder
        anomaly_detected = False
        anomaly_score = 0.0
        
        if autoencoder is not None and scaler is not None:
            try:
                input_data = np.array([[moisture, temperature]])
                input_scaled = scaler.transform(input_data)
                reconstruction = autoencoder.predict(input_scaled, verbose=0)
                mse = np.mean(np.power(input_scaled - reconstruction, 2))
                anomaly_score = float(mse)
                
                # Use configured threshold multiplier to reduce false positives
                adjusted_threshold = anomaly_threshold * config.THRESHOLD_MULTIPLIER
                anomaly_detected = mse > adjusted_threshold
                
                if anomaly_detected and config.PRINT_ANOMALY_ALERTS:
                    print(f"⚠️  ANOMALY DETECTED! Score: {anomaly_score:.6f} (Threshold: {adjusted_threshold:.6f})")
                    print(f"   Moisture: {moisture:.2f}%, Temperature: {temperature:.2f}°C")
                
            except Exception as e:
                print(f"Error in anomaly detection: {e}")
        
        current_readings = {
            'moisture': round(float(moisture), 2),
            'temperature': round(float(temperature), 2),
            'valve': int(valve),
            'timestamp': datetime.now().isoformat(),
            'anomaly_detected': bool(anomaly_detected),
            'anomaly_score': round(anomaly_score, 6)
        }
        
        time.sleep(config.UPDATE_INTERVAL)  # Update based on config

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SMART IRRIGATION ANOMALY DETECTION SYSTEM")
    print("="*60)
    
    # Load the trained model
    load_trained_model()
    
    # Start sensor update thread
    sensor_thread = threading.Thread(target=update_sensor_readings, daemon=True)
    sensor_thread.start()
    
    print("\n🚀 Starting Flask server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)