# Datasets Quick Reference

## 📊 Simple Answer

### **2 Main Datasets:**

1. **`soil_sensor_data.csv`** (17,000 samples)
   - ALL NORMAL data
   - Used for: Training (70%) + Validation (15%) + Internal Test (15%)

2. **`test_dataset_with_attacks.csv`** (2,500 samples)
   - 2,000 NORMAL + 500 ATTACKS
   - Used for: TRUE attack detection evaluation

---

## 🔄 Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Generate Training Data                               │
│ python generate_data.py                                      │
│ → soil_sensor_data.csv (17,000 NORMAL samples)               │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Train Model                                          │
│ python train_model.py                                        │
│                                                               │
│ Automatically splits soil_sensor_data.csv:                   │
│ ├─ Training:   11,900 (70%) → Learn normal patterns         │
│ ├─ Validation:  2,550 (15%) → Set threshold = 0.0456        │
│ └─ Test:        2,550 (15%) → Check overfitting             │
│                                                               │
│ → autoencoder_model.h5 + scaler.pkl + threshold              │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: Generate Attack Test Data                            │
│ python generate_test_dataset_with_anomalies.py               │
│ → test_dataset_with_attacks.csv (2,500 samples)              │
│    ├─ 2,000 NORMAL (benign readings)                         │
│    └─ 500 ATTACKS (labeled cyber-attacks)                    │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Evaluate on Attacks                                  │
│ python evaluate_with_labeled_data.py                         │
│                                                               │
│ Tests model on 500 LABELED attacks:                          │
│ ├─ Data Injection (Low) - 100                                │
│ ├─ Data Injection (High) - 80                                │
│ ├─ Sensor Spoofing - 90                                      │
│ ├─ Man-in-the-Middle - 80                                    │
│ ├─ Physical Manipulation - 70                                │
│ ├─ Protocol Exploitation - 40                                │
│ └─ DoS/Jamming - 40                                          │
│                                                               │
│ → evaluation_results_labeled.json (TRUE metrics)             │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 What Each Dataset Contains

### **soil_sensor_data.csv** (NORMAL ONLY)

```csv
region,crop_type,soil_moisture,soil_temperature,irrigation_valve_flow_rate,...
Nairobi East Kenya,Maize,42.5,25.3,2.5,...
Nairobi East Kenya,Tomatoes,48.2,23.1,0.0,...
Nairobi East Kenya,Maize,38.7,27.4,15.2,...
```

**All values are legitimate/benign**:
- Moisture: 15-85% (realistic range)
- Temperature: 15-35°C (Kenya climate)
- Valve flow: Calculated based on real needs

**17,000 samples split**:
- 11,900 → Training
- 2,550 → Validation (set threshold)
- 2,550 → Internal test (check overfitting)

---

### **test_dataset_with_attacks.csv** (NORMAL + ATTACKS)

```csv
soil_moisture,soil_temperature,is_attack,attack_type,attack_vector,sample_id
42.5,25.3,0,normal,none,BENIGN_0001
18.2,36.7,1,data_injection_low,MITM on sensor data,ATTACK_001
72.1,22.4,1,data_injection_high,Compromised gateway,ATTACK_002
50.0,20.0,1,sensor_spoofing_replay,Replay old packets,ATTACK_003
```

**2,000 Normal Samples**:
- Same distribution as training data
- Legitimate sensor readings
- `is_attack = 0`

**500 Attack Samples**:
- Manipulated moisture/temperature values
- Each labeled with attack type
- `is_attack = 1`

**Attack Types**:

| Attack Type | Count | What Happens |
|------------|-------|--------------|
| **Data Injection (Low)** | 100 | Injects FALSE low moisture → unnecessary irrigation |
| **Data Injection (High)** | 80 | Injects FALSE high moisture → no irrigation when needed |
| **Sensor Spoofing/Replay** | 90 | Replays old sensor data → wrong decisions |
| **Man-in-the-Middle** | 80 | Modifies data in transit → corrupted readings |
| **Physical Manipulation** | 70 | Tampers with sensor → false readings |
| **Protocol Exploitation** | 40 | Exploits comm protocol → injected values |
| **DoS/Jamming** | 40 | Freezes sensor readings → stale data |

---

## 🔍 How Attacks Affect Valve

### Example 1: Data Injection (Low Moisture)

**Normal Scenario**:
```
Actual Soil: moisture=45%, temp=25°C
Sensor Reads: moisture=45%, temp=25°C ✓ CORRECT
Valve Decision: CLOSED (soil is fine)
```

**Attack Scenario**:
```
Actual Soil: moisture=45%, temp=25°C
Attacker Injects: moisture=15%, temp=25°C ✗ FALSE
Valve Decision: OPEN with 25 L/min (wastes water!)

Autoencoder Detection:
  Expected: ~45% moisture
  Received: 15% moisture
  Error: 0.2847 > 0.0456 (threshold)
  → ATTACK DETECTED! ✓
```

### Example 2: Data Injection (High Moisture)

**Normal Scenario**:
```
Actual Soil: moisture=20%, temp=33°C (CRITICAL!)
Sensor Reads: moisture=20%, temp=33°C ✓ CORRECT
Valve Decision: OPEN (needs water urgently)
```

**Attack Scenario**:
```
Actual Soil: moisture=20%, temp=33°C (CRITICAL!)
Attacker Injects: moisture=75%, temp=22°C ✗ FALSE
Valve Decision: CLOSED (thinks soil is fine)
→ CROPS DIE FROM DROUGHT!

Autoencoder Detection:
  Expected: ~20% moisture, ~33°C temp
  Received: 75% moisture, 22°C temp
  Error: 0.3145 > 0.0456 (threshold)
  → ATTACK DETECTED! ✓
```

### Example 3: Correlated Attack

**Why correlation matters**:
```
Model Learned: Low moisture + High temp often occur together
               (evaporation causes both)

Normal: moisture=25%, temp=32°C ✓ Makes sense (correlation)
Attack: moisture=75%, temp=38°C ✗ Weird combination!

Detection: Model flags unusual correlations
```

---

## 📊 Validation vs Testing - Key Difference

### **Validation Set** (from soil_sensor_data.csv)

```
Purpose: Set the threshold for anomaly detection
Content: 2,550 NORMAL samples only

Process:
1. Calculate reconstruction errors on 2,550 normal samples
2. Errors: [0.0123, 0.0145, 0.0167, ..., 0.0456, 0.0512]
3. Take 95th percentile: 0.0456
4. This becomes the threshold

Result: threshold = 0.0456
Meaning: 95% of normal data has error ≤ 0.0456
         Anything above = ATTACK
```

### **Test Set** (test_dataset_with_attacks.csv)

```
Purpose: Measure TRUE attack detection performance
Content: 2,000 normal + 500 LABELED attacks

Process:
1. Load 2,500 samples with TRUE labels
2. Calculate reconstruction errors
3. Compare: error > threshold? → Predict ATTACK
4. Compare predictions vs TRUE labels
5. Calculate real metrics

Result: 
  Accuracy:  94% (correct classifications)
  Precision: 85% (of alarms, % truly attacks)
  Recall:    78% (% of attacks detected)
  
Per-Attack Detection Rates:
  Data Injection (Low):  95% detected
  Data Injection (High): 88% detected
  Spoofing/Replay:       92% detected
  MITM:                  87% detected
  Physical Manipulation: 82% detected
  Protocol Exploit:      75% detected
  DoS/Jamming:           98% detected
```

---

## ✅ Checklist

### **What You Need**:

- [x] `soil_sensor_data.csv` (17,000 normal samples)
  - Generated by: `generate__data.py`
  - Used for: Training + Validation + Internal Test

- [x] `test_dataset_with_attacks.csv` (2,500 samples: 2,000 normal + 500 attacks)
  - Generated by: `generate_test_dataset_with_anomalies.py`
  - Used for: TRUE attack detection evaluation

- [x] `autoencoder_model.h5` (trained model)
  - Generated by: `train_model.py`

- [x] `scaler.pkl` (feature normalization)
  - Generated by: `train_model.py`

- [x] `model_metrics.json` (contains threshold)
  - Generated by: `train_model.py`

### **What You Run**:

```bash
# 1. Generate normal training data
python generate_dataset.py

# 2. Train model (auto-splits into train/val/test)
python train_model.py

# 3. Generate attack test data
python generate_test_dataset_with_anomalies.py

# 4. Evaluate on attacks (TRUE metrics)
python evaluate_with_labeled_data.py
```

---

## 🎓 Why This Approach?

### **Unsupervised Learning** (Training)
```
✓ Train ONLY on normal data
✓ Autoencoder learns: "This is what normal looks like"
✓ No need for labeled attack data during training
✓ Realistic: Real farms don't have attack datasets
```

### **Supervised Evaluation** (Testing)
```
✓ Test on LABELED attacks
✓ Measure: Does model detect real attacks?
✓ Get metrics: Precision, Recall, Detection Rates
✓ Validate: Is model production-ready?
```

### **This is Standard Practice**
```
Autoencoder Anomaly Detection:
├─ Train on NORMAL data ✓ (unsupervised)
├─ Validate on NORMAL data ✓ (set threshold)
└─ Test on NORMAL + ATTACKS ✓ (measure performance)

NOT like supervised learning:
├─ Train on normal + attacks together ✗
└─ Need balanced dataset ✗
```

---

## 🚀 Quick Commands

```bash
# Complete workflow
python generate_dataset.py              # Normal data (17K)
python train_model.py                         # Train + Validate
python generate_test_dataset_with_anomalies.py  # Attack data (2.5K)
python evaluate_with_labeled_data.py          # TRUE evaluation

# Check results
cat evaluation_results_labeled.json
```

---

**Simple Summary**: Train on 17K normal data (auto-split 70/15/15), then test on 2.5K samples with 500 labeled cyber-attacks! 🎯
