"""
Generate TEST dataset focused on SENSOR SPOOFING/INJECTION/MANIPULATION attacks

This creates realistic cyber-attack scenarios for smart irrigation systems:
1. Normal test samples (from same distribution as training)
2. Labeled cyber-attack anomalies (spoofing, injection, manipulation)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(123)  # Different seed from training data

print("="*70)
print("GENERATING TEST DATASET - SENSOR ATTACK FOCUSED")
print("="*70)

# Load training statistics to ensure test is from same distribution
try:
    import json
    with open('model_metrics.json', 'r') as f:
        training_stats = json.load(f)['feature_statistics']
    
    TRAIN_MOISTURE_MEAN = training_stats['moisture_mean']
    TRAIN_MOISTURE_STD = training_stats['moisture_std']
    TRAIN_TEMP_MEAN = training_stats['temperature_mean']
    TRAIN_TEMP_STD = training_stats['temperature_std']
    
    print(f"\n✓ Loaded training distribution statistics:")
    print(f"  Moisture: μ={TRAIN_MOISTURE_MEAN:.2f}%, σ={TRAIN_MOISTURE_STD:.2f}%")
    print(f"  Temperature: μ={TRAIN_TEMP_MEAN:.2f}°C, σ={TRAIN_TEMP_STD:.2f}°C")
except:
    print("\n⚠ Could not load training stats. Using default Kenya values.")
    TRAIN_MOISTURE_MEAN = 42.0
    TRAIN_MOISTURE_STD = 10.0
    TRAIN_TEMP_MEAN = 25.0
    TRAIN_TEMP_STD = 3.5

# Test set configuration
NORMAL_SAMPLES = 2000
ATTACK_SAMPLES = 500
TOTAL_SAMPLES = NORMAL_SAMPLES + ATTACK_SAMPLES

print(f"\n📊 Test Set Composition:")
print(f"  Normal samples:   {NORMAL_SAMPLES} ({100*NORMAL_SAMPLES/TOTAL_SAMPLES:.1f}%)")
print(f"  Attack samples:   {ATTACK_SAMPLES} ({100*ATTACK_SAMPLES/TOTAL_SAMPLES:.1f}%)")
print(f"  Total samples:    {TOTAL_SAMPLES}")

# Define SENSOR ATTACK TYPES (Cyber-security focused)
ATTACK_TYPES = {
    'data_injection_low': {
        'description': 'Attacker injecting false LOW moisture readings to waste water',
        'count': 100,
        'attack_vector': 'Man-in-the-Middle (MITM) on sensor data',
        'moisture_manipulation': lambda: np.random.uniform(-30, -15),  # Inject low values
        'temp_manipulation': lambda: np.random.uniform(-3, 3),  # Minimal temp change
        'motivation': 'Waste water resources, increase irrigation costs'
    },
    'data_injection_high': {
        'description': 'Attacker injecting false HIGH moisture to prevent irrigation',
        'count': 80,
        'attack_vector': 'Compromised IoT gateway, data injection',
        'moisture_manipulation': lambda: np.random.uniform(15, 35),  # Inject high values
        'temp_manipulation': lambda: np.random.uniform(-5, -2),  # Lower temp
        'motivation': 'Cause crop failure through drought'
    },
    'sensor_spoofing_replay': {
        'description': 'Replay attack - sending old sensor readings',
        'count': 90,
        'attack_vector': 'Replay previous sensor data packets',
        'moisture_manipulation': lambda: np.random.choice([5, 10, 15, 20]) - np.random.uniform(0, 3),  # Fixed old values
        'temp_manipulation': lambda: np.random.choice([20, 22, 24, 26]) - TRAIN_TEMP_MEAN,  # Fixed old temps
        'motivation': 'Hide actual field conditions, prevent proper irrigation'
    },
    'man_in_the_middle_modify': {
        'description': 'MITM attack modifying sensor data in transit',
        'count': 80,
        'attack_vector': 'Intercept and modify packets between sensor and gateway',
        'moisture_manipulation': lambda: np.random.uniform(-20, -10),  # Decrease moisture
        'temp_manipulation': lambda: np.random.uniform(8, 15),  # Increase temperature
        'motivation': 'Trigger excessive irrigation, sabotage'
    },
    'sensor_value_manipulation': {
        'description': 'Direct sensor tampering - physical manipulation',
        'count': 70,
        'attack_vector': 'Physical access to sensor, rewiring',
        'moisture_manipulation': lambda: np.random.uniform(-25, -12),  # Random manipulation
        'temp_manipulation': lambda: np.random.uniform(6, 12),  # Random manipulation
        'motivation': 'Sabotage, competitive advantage'
    },
    'protocol_exploitation': {
        'description': 'Exploiting communication protocol vulnerabilities',
        'count': 40,
        'attack_vector': 'Buffer overflow, command injection on IoT protocol',
        'moisture_manipulation': lambda: np.random.uniform(-15, -5),  # Moderate changes
        'temp_manipulation': lambda: np.random.uniform(5, 10),  # Moderate changes
        'motivation': 'System compromise, data theft'
    },
    'dos_sensor_jamming': {
        'description': 'Denial of Service - sensor readings stuck/frozen',
        'count': 40,
        'attack_vector': 'RF jamming, communication disruption',
        'moisture_manipulation': lambda: np.random.choice([10, 20, 30, 40, 50]) - TRAIN_MOISTURE_MEAN,  # Frozen values
        'temp_manipulation': lambda: np.random.choice([20, 25, 30]) - TRAIN_TEMP_MEAN,  # Frozen values
        'motivation': 'Prevent monitoring, hide theft'
    }
}

# Verify counts
total_attacks = sum(spec['count'] for spec in ATTACK_TYPES.values())
assert total_attacks == ATTACK_SAMPLES, f"Attack counts ({total_attacks}) don't sum to {ATTACK_SAMPLES}"

print(f"\n" + "="*70)
print("SENSOR ATTACK TYPES (Cyber-Security Focus)")
print("="*70)
for attack_type, spec in ATTACK_TYPES.items():
    print(f"\n🔴 {attack_type.upper()}")
    print(f"   Description:    {spec['description']}")
    print(f"   Attack Vector:  {spec['attack_vector']}")
    print(f"   Samples:        {spec['count']} ({100*spec['count']/ATTACK_SAMPLES:.1f}%)")
    print(f"   Motivation:     {spec['motivation']}")

# Generate normal test samples
print(f"\n" + "="*70)
print(f"GENERATING DATA")
print("="*70)
print(f"\n🔄 Generating {NORMAL_SAMPLES} normal (benign) samples...")
normal_data = []

for i in range(NORMAL_SAMPLES):
    # Generate from same distribution as training (legitimate sensor readings)
    moisture = np.random.normal(TRAIN_MOISTURE_MEAN, TRAIN_MOISTURE_STD)
    temperature = np.random.normal(TRAIN_TEMP_MEAN, TRAIN_TEMP_STD)
    
    # Ensure realistic bounds for legitimate data
    moisture = max(15, min(85, moisture))
    temperature = max(15, min(35, temperature))
    
    normal_data.append({
        'soil_moisture': round(moisture, 2),
        'soil_temperature': round(temperature, 2),
        'is_attack': 0,
        'attack_type': 'normal',
        'attack_vector': 'none',
        'sample_id': f"BENIGN_{i:04d}"
    })

# Generate attack samples
print(f"\n🚨 Generating {ATTACK_SAMPLES} cyber-attack samples...")
attack_data = []
sample_id = 0

for attack_type, spec in ATTACK_TYPES.items():
    print(f"  - Generating {spec['count']} {attack_type} attacks...")
    for i in range(spec['count']):
        # Start with legitimate baseline readings
        base_moisture = np.random.normal(TRAIN_MOISTURE_MEAN, TRAIN_MOISTURE_STD)
        base_temp = np.random.normal(TRAIN_TEMP_MEAN, TRAIN_TEMP_STD)
        
        # Apply attack manipulation
        moisture = base_moisture + spec['moisture_manipulation']()
        temperature = base_temp + spec['temp_manipulation']()
        
        # For attacks, allow out-of-normal-range values
        # But keep physically possible bounds
        moisture = max(-5, min(120, moisture))  # Can go slightly negative (sensor error)
        temperature = max(-5, min(60, temperature))
        
        attack_data.append({
            'soil_moisture': round(moisture, 2),
            'soil_temperature': round(temperature, 2),
            'is_attack': 1,
            'attack_type': attack_type,
            'attack_vector': spec['attack_vector'],
            'sample_id': f"ATTACK_{attack_type.upper()}_{i:04d}"
        })
        sample_id += 1

# Combine and shuffle
all_data = normal_data + attack_data
np.random.shuffle(all_data)

# Create DataFrame
df = pd.DataFrame(all_data)

# Statistics
print("\n" + "="*70)
print("TEST DATASET STATISTICS")
print("="*70)

print(f"\n📊 Overall Composition:")
print(f"  Total samples:     {len(df)}")
print(f"  Benign (normal):   {(df['is_attack']==0).sum()} ({100*(df['is_attack']==0).sum()/len(df):.1f}%)")
print(f"  Attacks (malicious): {(df['is_attack']==1).sum()} ({100*(df['is_attack']==1).sum()/len(df):.1f}%)")

print(f"\n📊 Feature Statistics:")
print(f"\nBenign Samples (Legitimate Sensor Readings):")
benign_df = df[df['is_attack'] == 0]
print(f"  Moisture:    Mean={benign_df['soil_moisture'].mean():.2f}%, Std={benign_df['soil_moisture'].std():.2f}%")
print(f"               Range=[{benign_df['soil_moisture'].min():.2f}, {benign_df['soil_moisture'].max():.2f}]")
print(f"  Temperature: Mean={benign_df['soil_temperature'].mean():.2f}°C, Std={benign_df['soil_temperature'].std():.2f}°C")
print(f"               Range=[{benign_df['soil_temperature'].min():.2f}, {benign_df['soil_temperature'].max():.2f}]")

print(f"\nAttack Samples (Manipulated/Spoofed Readings):")
attack_df = df[df['is_attack'] == 1]
print(f"  Moisture:    Mean={attack_df['soil_moisture'].mean():.2f}%, Std={attack_df['soil_moisture'].std():.2f}%")
print(f"               Range=[{attack_df['soil_moisture'].min():.2f}, {attack_df['soil_moisture'].max():.2f}]")
print(f"  Temperature: Mean={attack_df['soil_temperature'].mean():.2f}°C, Std={attack_df['soil_temperature'].std():.2f}°C")
print(f"               Range=[{attack_df['soil_temperature'].min():.2f}, {attack_df['soil_temperature'].max():.2f}]")

print(f"\n📊 Attack Type Distribution:")
attack_counts = df[df['is_attack']==1]['attack_type'].value_counts()
for atk_type, count in attack_counts.items():
    print(f"  {atk_type:30} - {count:3d} samples ({100*count/attack_df.shape[0]:5.1f}%)")

# Sample records
print("\n" + "="*70)
print("SAMPLE RECORDS")
print("="*70)

print("\n✓ Benign Samples (Legitimate - first 5):")
print(df[df['is_attack']==0].head(5)[['soil_moisture', 'soil_temperature', 'attack_type', 'sample_id']].to_string(index=False))

print("\n🔴 Attack Samples (5 random from different types):")
sample_attacks = []
for atk_type in df[df['is_attack']==1]['attack_type'].unique()[:5]:
    sample = df[df['attack_type']==atk_type].iloc[0]
    sample_attacks.append(sample)
sample_df = pd.DataFrame(sample_attacks)
print(sample_df[['soil_moisture', 'soil_temperature', 'attack_type', 'attack_vector']].to_string(index=False))

# Save to CSV
output_file = 'test_dataset_with_attacks.csv'
df.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ CYBER-ATTACK TEST DATASET GENERATED!")
print("="*70)
print(f"\n✓ Saved to: {output_file}")
print(f"✓ Total samples: {len(df)}")
print(f"✓ Benign (legitimate): {(df['is_attack']==0).sum()}")
print(f"✓ Attacks (malicious): {(df['is_attack']==1).sum()}")

print("\n" + "="*70)
print("ATTACK SCENARIO SUMMARY")
print("="*70)
print("\nThis test set simulates realistic IoT/ICS cyber-attacks:")
print("  1. Data Injection (Low/High) - False sensor data injected")
print("  2. Replay Attacks - Old data packets replayed")
print("  3. Man-in-the-Middle - Data modified in transit")
print("  4. Physical Tampering - Direct sensor manipulation")
print("  5. Protocol Exploitation - Buffer overflow, command injection")
print("  6. DoS/Jamming - Communication disruption, frozen values")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. This dataset contains LABELED cyber-attacks for ICS/IoT security testing")
print("2. Run: python evaluate_with_labeled_data.py")
print("3. Get TRUE detection rates for each attack type")
print("4. Analyze which attacks are easier/harder to detect")
print("5. Tune threshold based on attack detection requirements")
print("\n" + "="*70)

# Save attack type descriptions for reference
attack_info = {
    'dataset_info': {
        'total_samples': len(df),
        'benign_samples': int((df['is_anomaly']==0).sum()),
        'attack_samples': int((df['is_anomaly']==1).sum()),
        'focus': 'Sensor Spoofing, Injection, and Manipulation Attacks'
    },
    'attack_types': {}
}

for anomaly_type, spec in ATTACK_TYPES.items():
    attack_info['attack_types'][anomaly_type] = {
        'description': spec['description'],
        'attack_vector': spec['attack_vector'],
        'motivation': spec['motivation'],
        'count': spec['count']
    }

with open('attack_types_info.json', 'w') as f:
    json.dump(attack_info, f, indent=2)

print("\n✓ Attack type descriptions saved to: attack_types_info.json")
print("\n" + "="*70)