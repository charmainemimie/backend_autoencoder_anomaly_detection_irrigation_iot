# routes/simulate_attack.py
from flask import Blueprint, request, jsonify
import numpy as np
from datetime import datetime

simulate_attack_bp = Blueprint('simulate_attack', __name__)

def simulate_attack(attack_type, data):
    """
    Simulates various cyberattacks on soil sensor data.
    data: dictionary containing moisture/temp/nutrients
    """
    attacked = data.copy()

    if attack_type == "data_replay":
        # Replay old moisture values (classic IoT attack)
        attacked["moisture"] = np.random.choice([10, 20, 30])
        attacked["note"] = "Replay attack: moisture values from past used."

    elif attack_type == "random_noise":
        for key in attacked:
            if key != "note":
                attacked[key] += np.random.uniform(-20, 20)
        attacked["note"] = "Random noise injection attack."

    elif attack_type == "spoofing":
        attacked["temperature"] += 15      # Fake overheating
        attacked["moisture"] -= 20         # Pretend drought
        attacked["note"] = "Sensor spoofing: values intentionally falsified."

    elif attack_type == "drift_attack":
        # Slowly push values toward dangerous ranges
        attacked["moisture"] += np.random.uniform(5, 10)
        attacked["temperature"] += np.random.uniform(3, 6)
        attacked["note"] = "Concept drift attack: slow subtle manipulation."

    elif attack_type == "constant_value":
        for key in attacked:
            if key != "note":
                attacked[key] = 50
        attacked["note"] = "Constant value attack: sensor stuck at 50."

    else:
        attacked["note"] = "Unknown attack type."

    return attacked


@simulate_attack_bp.route('/api/simulate_attack', methods=['POST'])
def simulate_attack_route():
    """
    Input:
    {
        "attack_type": "spoofing",
        "region": "kiambu"
    }
    """
    body = request.json
    attack_type = body.get("attack_type", "spoofing")
    region = body.get("region", "kiambu")

    from app import models, features

    if region not in models:
        return jsonify({"error": "Invalid region"}), 400

    # Create baseline normal data
    normal_data = {
        "moisture": np.random.uniform(30, 70),
        "temperature": np.random.uniform(15, 35),
        "nitrogen": np.random.uniform(20, 80),
        "phosphorus": np.random.uniform(10, 60),
        "potassium": np.random.uniform(20, 70),
    }

    # Generate malicious version
    attacked = simulate_attack(attack_type, normal_data)

    # Convert to array for model
    X = np.array([
        attacked["moisture"],
        attacked["temperature"],
        attacked["nitrogen"],
        attacked["phosphorus"],
        attacked["potassium"],
    ]).reshape(1, -1)

    model = models[region]
    prediction = model.predict(X)[0]
    raw_score = model.decision_function(X)[0]
    anomaly_score = max(0, min(1, 0.5 - raw_score))

    return jsonify({
        "region": region,
        "timestamp": datetime.now().isoformat(),
        "attack_type": attack_type,
        "original_data": normal_data,
        "attacked_data": attacked,
        "is_anomaly": prediction == -1 or anomaly_score > 0.6,
        "anomaly_score": float(anomaly_score),
        "prediction": int(prediction)
    })
