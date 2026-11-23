# 🌱 AgriCyber - Smart Irrigation Anomaly Detection System

<div align="center">

**Autoencoder-Based Cybersecurity for IoT-Enabled Agricultural Irrigation Systems**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

_Protecting East African Smart Agriculture from Cyber Threats_

</div>

## 🎬 Demo Video

[Watch the demo video](https://github.com/user-attachments/assets/0b7ab90a-eb44-4d61-81c5-95333e851ab0)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [The Problem](#-the-problem)
- [Solution Architecture](#-solution-architecture)
- [Machine Learning Model](#-machine-learning-model)
- [Dataset & Data Collection](#-dataset--data-collection)
- [Cyber-Attack Detection](#-cyber-attack-detection)
- [Performance Metrics](#-performance-metrics)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [API Endpoints](#-api-endpoints)
- [Configuration & Tuning](#-configuration--tuning)
- [Deployment](#-deployment)
- [Project Documentation](#-project-documentation)
- [Future Work](#-future-work)
- [Contributors](#-contributors)

---

## 🌍 Project Overview

**AgriCyber** is an innovative cybersecurity solution designed to protect IoT-enabled smart irrigation systems against cyber-attacks. This project represents the **backend component** of a comprehensive agricultural cybersecurity platform that uses **deep learning-based anomaly detection** to identify malicious interference with soil sensors in real-time.

The system leverages an **Autoencoder neural network** trained exclusively on normal sensor data to detect deviations caused by various cyber-attack vectors, including data injection, sensor spoofing, man-in-the-middle attacks, and physical sensor manipulation.

### 🎯 Key Objectives

1. **Real-Time Anomaly Detection**: Identify cyber-attacks on soil moisture and temperature sensors within seconds
2. **Water Resource Protection**: Prevent water waste caused by malicious manipulation of sensor readings
3. **Crop Protection**: Safeguard crops from damage due to fraudulent denial of irrigation
4. **Agricultural Cybersecurity**: Provide a robust defense layer for smart farming infrastructure

---

## 🚨 The Problem

### Smart Agriculture Security Gap

Modern precision agriculture relies heavily on IoT sensors to automate irrigation decisions. These sensors measure:

- **Soil Moisture** (%) - Determines when crops need water
- **Soil Temperature** (°C) - Triggers cooling irrigation during heat stress

However, these IoT systems are vulnerable to **cyber-attacks** that can:

| Attack Scenario         | Method                                | Consequence                                            |
| ----------------------- | ------------------------------------- | ------------------------------------------------------ |
| **False Low Moisture**  | Inject fake readings showing dry soil | Valve opens unnecessarily → **Water waste**            |
| **False High Moisture** | Inject fake readings showing wet soil | Valve stays closed → **Crop drought damage**           |
| **Replay Attack**       | Send old sensor data                  | System makes outdated decisions → **Wrong irrigation** |
| **MITM Modification**   | Alter data in transit                 | Corrupted readings → **Unpredictable behavior**        |

### Real-World Impact

In agriculture-dependent regions like East Africa, such attacks could cause:

- 💧 Significant water resource depletion
- 🌾 Crop failures affecting food security
- 💰 Economic losses for farmers
- 🚜 Loss of trust in smart agriculture technology

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AgriCyber System Architecture                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐     ┌────────────────────┐     ┌────────────────┐ │
│  │   IoT Sensors    │────▶│  Flask Backend API  │────▶│   Dashboard    │ │
│  │ (Moisture/Temp)  │     │                    │     │   Frontend     │ │
│  └──────────────────┘     └─────────┬──────────┘     └────────────────┘ │
│                                     │                                    │
│                           ┌─────────▼──────────┐                        │
│                           │   Autoencoder      │                        │
│                           │   Anomaly Detector │                        │
│                           │                    │                        │
│                           │  ┌──────────────┐  │                        │
│                           │  │   Encoder    │  │                        │
│                           │  │ (8→4 neurons)│  │                        │
│                           │  └──────┬───────┘  │                        │
│                           │         │          │                        │
│                           │  ┌──────▼───────┐  │                        │
│                           │  │   Decoder    │  │                        │
│                           │  │ (4→8→2)      │  │                        │
│                           │  └──────────────┘  │                        │
│                           └────────────────────┘                        │
│                                     │                                    │
│                           ┌─────────▼──────────┐                        │
│                           │  Reconstruction    │                        │
│                           │  Error Analysis    │                        │
│                           │                    │                        │
│                           │  Error > Threshold │                        │
│                           │        ↓           │                        │
│                           │  🚨 ANOMALY ALERT  │                        │
│                           └────────────────────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component            | Description                                                 | Technology                  |
| -------------------- | ----------------------------------------------------------- | --------------------------- |
| **Backend API**      | RESTful API for sensor data ingestion and anomaly detection | Flask, Python               |
| **ML Engine**        | Autoencoder model for anomaly detection                     | TensorFlow/Keras            |
| **Data Pipeline**    | Real-time sensor data processing and scaling                | NumPy, Pandas, Scikit-learn |
| **Attack Simulator** | Testing module for simulating cyber-attacks                 | Custom Python               |

---

## 🧠 Machine Learning Model

### Autoencoder Architecture

The core detection engine is a **Deep Autoencoder** neural network trained using **unsupervised learning** to reconstruct normal sensor patterns.

```
                    INPUT                          OUTPUT
                 [moisture,                     [reconstructed
                temperature]                 moisture, temp]
                     │                              ▲
                     ▼                              │
              ┌────────────┐                 ┌────────────┐
              │  Dense(8)  │                 │  Dense(2)  │
              │    ReLU    │                 │   Linear   │
              │ Dropout(0.1)│                │            │
              └─────┬──────┘                 └─────┬──────┘
                    │                              │
                    ▼                              │
              ┌────────────┐                 ┌────────────┐
              │  Dense(4)  │ ──────────────▶ │  Dense(8)  │
              │    ReLU    │    LATENT       │    ReLU    │
              │  (Encoded) │    SPACE        │ Dropout(0.1)│
              └────────────┘                 └────────────┘
                 ENCODER                        DECODER
```

### Model Specifications

| Parameter            | Value                                            |
| -------------------- | ------------------------------------------------ |
| **Input Features**   | 2 (soil_moisture, soil_temperature)              |
| **Encoder Layers**   | Dense(8, ReLU) → Dropout(0.1) → Dense(4, ReLU)   |
| **Decoder Layers**   | Dense(8, ReLU) → Dropout(0.1) → Dense(2, Linear) |
| **Loss Function**    | Mean Squared Error (MSE)                         |
| **Optimizer**        | Adam                                             |
| **Training Epochs**  | 100                                              |
| **Batch Size**       | 32                                               |
| **Validation Split** | 10% of training data                             |

### How Detection Works

1. **Training Phase**: Model learns to reconstruct **normal** sensor readings accurately
2. **Inference Phase**: New readings are passed through the autoencoder
3. **Error Calculation**: Compute MSE between input and reconstruction
4. **Threshold Comparison**: If error > threshold → **ANOMALY DETECTED**

```python
# Detection Algorithm
reconstruction = autoencoder.predict(scaled_input)
error = mean_squared_error(scaled_input, reconstruction)

if error > threshold:
    alert("🚨 CYBER-ATTACK DETECTED!")
```

---

## 📊 Dataset & Data Collection

### Training Dataset: `soil_sensor_data.csv`

A synthetic dataset simulating **17,000 normal sensor readings** from a smart irrigation system in Nairobi East, Kenya.

#### Dataset Characteristics

| Attribute         | Value                                            |
| ----------------- | ------------------------------------------------ |
| **Location**      | Nairobi East, Kenya                              |
| **Total Records** | 17,000 samples                                   |
| **Time Period**   | Full year simulation with seasonal variations    |
| **Crops Covered** | Maize (8,500 records), Tomatoes (8,500 records)  |
| **Data Sources**  | Simulated based on Kenyan agricultural standards |

#### Features

| Feature                      | Description                         | Range                         |
| ---------------------------- | ----------------------------------- | ----------------------------- |
| `soil_moisture`              | Volumetric water content (%)        | 15% - 85%                     |
| `soil_temperature`           | Soil temperature (°C)               | 15°C - 35°C                   |
| `irrigation_valve_flow_rate` | Water flow when valve opens (L/min) | 0 - 50 L/min                  |
| `irrigation_type`            | Type of irrigation system           | Drip, Sprinkler, Manual, None |
| `crop_type`                  | Crop being monitored                | Maize, Tomatoes               |
| `timestamp`                  | Date of reading                     | DD/MM/YYYY                    |
| `sensor_id`                  | Unique sensor identifier            | SENSYN*KE001*\*               |

#### Data Distribution Statistics

```
Soil Moisture (%):
  ├── Mean:  51.58%
  ├── Std:   10.17%
  └── Range: [15% - 85%]

Soil Temperature (°C):
  ├── Mean:  24.60°C
  ├── Std:   5.17°C
  └── Range: [10°C - 42°C]
```

#### Realistic Variations Included

- 🌦️ **Seasonal Patterns**: Kenyan long rains (Mar-May) and short rains (Oct-Dec)
- 🌡️ **Diurnal Cycles**: Day/night temperature and moisture fluctuations
- 🌾 **Crop-Specific Thresholds**: Different irrigation requirements for Maize vs. Tomatoes
- 💧 **Flow Rate Variations**: Drip (2.5 L/min), Sprinkler (15 L/min), Manual (35 L/min)

---

### Test Dataset: `test_dataset_with_attacks.csv`

A separate evaluation dataset containing **2,500 labeled samples** for true performance measurement.

| Category          | Count | Percentage |
| ----------------- | ----- | ---------- |
| **Normal/Benign** | 2,000 | 80%        |
| **Cyber-Attacks** | 500   | 20%        |

---

## 🔴 Cyber-Attack Detection

### Supported Attack Types

The system is designed to detect **7 categories of IoT/ICS cyber-attacks**:

| Attack Type                | Description                               | Sample Count | Detection Rate |
| -------------------------- | ----------------------------------------- | ------------ | -------------- |
| **Data Injection (Low)**   | Fake low moisture readings to waste water | 100          | 40.0%          |
| **Data Injection (High)**  | Fake high moisture to prevent irrigation  | 80           | 66.3%          |
| **Sensor Spoofing/Replay** | Replaying old sensor data packets         | 90           | 18.9%          |
| **Man-in-the-Middle**      | Modifying data in transit                 | 80           | 53.8%          |
| **Physical Manipulation**  | Direct sensor tampering                   | 70           | 50.0%          |
| **Protocol Exploitation**  | Buffer overflow, command injection        | 40           | 27.5%          |
| **DoS/Sensor Jamming**     | RF jamming, frozen sensor values          | 40           | 42.5%          |

### Attack Vector Details

```json
{
  "data_injection_low": {
    "motivation": "Waste water resources, increase irrigation costs",
    "technique": "Man-in-the-Middle on sensor data stream",
    "effect": "Inject falsely LOW moisture → unnecessary valve opening"
  },
  "data_injection_high": {
    "motivation": "Cause crop failure through drought",
    "technique": "Compromised IoT gateway, data injection",
    "effect": "Inject falsely HIGH moisture → prevent needed irrigation"
  },
  "sensor_spoofing_replay": {
    "motivation": "Hide actual field conditions",
    "technique": "Replay previous sensor data packets",
    "effect": "System makes decisions based on outdated information"
  },
  "man_in_the_middle_modify": {
    "motivation": "Trigger excessive irrigation, sabotage",
    "technique": "Intercept and modify packets between sensor and gateway",
    "effect": "Corrupted readings lead to wrong valve decisions"
  }
}
```

---

## 📈 Performance Metrics

### Model Training Performance

| Metric         | Training | Testing  |
| -------------- | -------- | -------- |
| **Loss (MSE)** | 0.009385 | 0.009485 |
| **MAE**        | 0.07408  | 0.07403  |

### Anomaly Detection Threshold

```
Threshold = 95th percentile of training reconstruction errors
         = 0.035232
```

### Classification Metrics on Labeled Test Data

| Metric        | Value  |
| ------------- | ------ |
| **Accuracy**  | 83.44% |
| **Precision** | 62.43% |
| **Recall**    | 43.20% |
| **F1-Score**  | 0.5106 |
| **ROC-AUC**   | 0.8022 |
| **PR-AUC**    | 0.6063 |

### Confusion Matrix

```
                     Predicted Normal    Predicted Anomaly
Actual Normal             1,870               130
Actual Anomaly              284               216
```

### Error Distribution

| Sample Type | Mean Error | Std    | Separation Ratio |
| ----------- | ---------- | ------ | ---------------- |
| **Normal**  | 0.0101     | 0.0148 | -                |
| **Anomaly** | 0.0473     | 0.0528 | **4.70x**        |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or 3.11 (TensorFlow 2.15 and numpy 1.25 don't support 3.12+)
- pip package manager
- Virtual environment (recommended)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/charmainemimie/backend_autoencoder_anomaly_detection_irrigation_iot.git
cd backend_autoencoder_anomaly_detection_irrigation_iot

# 2. Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate training dataset
python generate_data.py

# 5. Train the model
python train_model.py

# 6. Start the backend server
python app.py
```

### Dependencies

```
Flask==2.3.3
flask-cors==3.0.10
gunicorn==21.2.0
numpy==1.25.2
pandas==2.1.4
scikit-learn==1.2.2
tensorflow-cpu==2.15.0
matplotlib==3.7.2
scipy==1.11.2
joblib==1.3.2
python-dotenv==1.0.0
```

---

## 📖 Usage Guide

### Complete Workflow

```bash
# Step 1: Generate Normal Training Data (17,000 samples)
python generate_data.py
# Output: soil_sensor_data.csv

# Step 2: Train the Autoencoder Model
python train_model.py
# Outputs: autoencoder_model.h5, autoencoder_weights.npz, scaler.pkl, model_metrics.json
# (app.py loads autoencoder_weights.npz and runs inference with numpy, without TensorFlow)

# Step 3: Generate Attack Test Dataset (optional, for evaluation)
python generate_test_data_with_anomalies.py
# Output: test_dataset_with_attacks.csv

# Step 4: Evaluate Model Performance (optional)
python evaluate_with_labeled_data.py
# Output: evaluation_results_labeled.json

# Step 5: Start the Backend Server
python app.py
# Server runs at http://localhost:5000
```

### Testing Attack Detection

The backend includes a built-in attack simulator accessible via API:

```bash
# Simulate moisture injection attack
curl -X POST http://localhost:5000/api/simulate-attack \
  -H "Content-Type: application/json" \
  -d '{"active": true, "target": "moisture", "intensity": 20}'

# Stop attack simulation
curl -X POST http://localhost:5000/api/simulate-attack \
  -H "Content-Type: application/json" \
  -d '{"active": false}'
```

---

## 🔌 API Endpoints

### Core Endpoints

| Endpoint               | Method | Description                                     |
| ---------------------- | ------ | ----------------------------------------------- |
| `/api/readings`        | GET    | Get current sensor readings with anomaly status |
| `/api/login`           | POST   | Authenticate user                               |
| `/api/simulate-attack` | POST   | Simulate cyber-attack for testing               |

### Response Example: `/api/readings`

```json
{
  "moisture": 42.35,
  "temperature": 25.12,
  "valve": 0,
  "timestamp": "2024-01-15T14:30:00.000000",
  "anomaly_detected": false,
  "anomaly_score": 0.008234
}
```

### Response Example: Anomaly Detected

```json
{
  "moisture": 15.0,
  "temperature": 38.5,
  "valve": 1,
  "timestamp": "2024-01-15T14:30:05.000000",
  "anomaly_detected": true,
  "anomaly_score": 0.2847
}
```

---

## 🔧 Configuration & Tuning

### Configuration File: `config.py`

```python
# Base sensor values (match training data)
BASE_MOISTURE = 42.0        # %
BASE_TEMPERATURE = 24.5     # °C

# Simulation variation
MOISTURE_STD = 2.0          # Natural variation
TEMPERATURE_STD = 1.5       # Natural variation

# Anomaly Detection Sensitivity
THRESHOLD_MULTIPLIER = 1.5  # Higher = fewer false positives

# Irrigation Valve Thresholds
MIN_MOISTURE_THRESHOLD = 30   # Open valve below this (%)
MAX_TEMPERATURE_THRESHOLD = 32  # Open valve above this (°C)

# System Settings
UPDATE_INTERVAL = 1         # Seconds between readings
PRINT_ANOMALY_ALERTS = True # Console logging
```

### Tuning Guide

**If too many false alarms:**

```python
THRESHOLD_MULTIPLIER = 2.0  # Increase tolerance
MOISTURE_STD = 1.5          # Reduce simulation variation
TEMPERATURE_STD = 1.0       # Reduce simulation variation
```

**If attacks are not detected:**

```python
THRESHOLD_MULTIPLIER = 1.0  # Use exact training threshold
# Consider retraining with more diverse data
```

---

## 🚀 Deployment

### Render.com (Recommended)

Live instance: https://agricyber.onrender.com (for example `GET /api/readings`; there is no route at `/`).

The project includes `render.yaml` for Blueprint deployment:

```yaml
services:
  - type: web
    name: agricyber-backend
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt && python generate_data.py && python train_model.py
    startCommand: gunicorn app:app --workers 1 --threads 4 --timeout 120
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.12
      - key: TF_CPP_MIN_LOG_LEVEL
        value: "2"
```

Notes:

- **Python version** comes from `PYTHON_VERSION` or the `.python-version` file. Render ignores `runtime.txt`, and its default Python is too new for TensorFlow 2.15.
- **The model is trained during the build**, because the dataset and model files are gitignored. This adds roughly 1–2 minutes to each build.
- **The web process doesn't import TensorFlow.** It runs the exported weights in numpy (about 150 MB of RAM instead of more than 500 MB), so it fits on the free plan.
- **Keep one worker.** Sensor readings and attack state live in process memory.
- **Manual setup:** if the service wasn't created from `render.yaml`, set the same build command, start command and `PYTHON_VERSION` under the service's Settings.
- Optionally set `DEMO_PASSWORD` to change the demo login password (default `agricyber123`).

### Heroku

Use the included `Procfile`:

```
web: gunicorn app:app --workers 1 --threads 4 --timeout 120
```

### Docker (Custom)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Train at build time: the dataset and model files are not in the repo
RUN python generate_data.py && python train_model.py
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4"]
```

---

## 📚 Project Documentation

For comprehensive technical documentation, theoretical background, and detailed analysis, please refer to:

📄 **[ProjectDocumentation.pdf](ProjectDocumentation.pdf)**

This document contains:

- Full literature review on IoT security in agriculture
- Detailed autoencoder architecture analysis
- Complete experimental methodology
- Extended performance evaluation
- Research references and citations
- Future research directions

---

## 🔮 Future Work

- [ ] **Multi-sensor fusion**: Incorporate additional sensors (pH, humidity, light)
- [ ] **LSTM Enhancement**: Add temporal pattern learning for sequence-based attacks
- [ ] **Federated Learning**: Distributed model training across multiple farms
- [ ] **Edge Deployment**: TensorFlow Lite model for on-device inference
- [ ] **Explainable AI**: Add SHAP/LIME explanations for anomaly alerts
- [ ] **Threshold Optimization**: Automated threshold tuning using Bayesian optimization

---

## 👥 Contributors

This project is part of the **AgriCyber Initiative** - an effort to bring cybersecurity best practices to smart agriculture systems in developing regions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- East Africa Agricultural IoT Research Community
- TensorFlow and Keras Development Teams
- Open-source contributors and reviewers

---

<div align="center">

**Protecting Smart Farms, One Sensor at a Time** 🌾🛡️

_Made with ❤️ for Agricultural Cybersecurity_

</div>
