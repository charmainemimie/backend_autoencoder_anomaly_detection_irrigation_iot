import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Farm details
FARM_REGION = "Nairobi East, Kenya"
SENSOR_ID_BASE = "SENSYN_KE001"

# Dataset parameters
TOTAL_RECORDS = 17000
START_DATE = datetime(2024, 1, 1, 6, 0, 0)  # Start January 1, 2024 at 6 AM

# Two crops with different moisture and temperature requirements
CROPS = {
    'Maize': {
        'optimal_moisture': 45.0,  # % - Maize needs moderate moisture
        'moisture_std': 8.0,
        'optimal_temp': 26.0,  # °C - Warm season crop
        'temp_std': 3.5,
        'valve_moisture_threshold': 35.0,  # Open valve below this
        'valve_temp_threshold': 32.0,  # Open valve above this (heat stress)
        'records': 8500  # Half the dataset
    },
    'Tomatoes': {
        'optimal_moisture': 55.0,  # % - Tomatoes need higher moisture
        'moisture_std': 7.0,
        'optimal_temp': 23.0,  # °C - Cooler preference
        'temp_std': 3.0,
        'valve_moisture_threshold': 45.0,  # Open valve below this (higher need)
        'valve_temp_threshold': 30.0,  # Open valve above this (more sensitive)
        'records': 8500  # Half the dataset
    }
}

# Irrigation system flow rates (based on industry standards)
# Source: Research shows typical small farm irrigation rates
IRRIGATION_FLOW_RATES = {
    'Drip': {
        'base_flow': 2.5,  # L/min - Typical drip emitters: 1-4 L/hour = 0.017-0.067 L/min per emitter
        'std': 0.8,        # Multiple emitters in a zone
        'description': 'Drip irrigation (efficient, low flow)'
    },
    'Sprinkler': {
        'base_flow': 15.0,  # L/min - Small farm sprinklers: 10-25 L/min typical
        'std': 3.5,
        'description': 'Sprinkler irrigation (moderate flow)'
    },
    'Manual': {
        'base_flow': 35.0,  # L/min - Hose or manual watering: 30-50 L/min
        'std': 8.0,
        'description': 'Manual/hose irrigation (high flow)'
    }
}

def calculate_valve_flow_and_time(moisture, temperature, crop_config):
    """
    Calculate irrigation valve flow rate (L/min) and duration based on soil conditions
    Returns: (flow_rate_L_min, duration_minutes, irrigation_type)
    
    Logic based on agricultural irrigation standards:
    - Drip: 1-4 L/hour per emitter, efficient for targeted watering
    - Sprinkler: 10-25 L/min for small farm systems
    - Manual: 30-50 L/min for hose/manual watering
    """
    moisture_deficit = max(0, crop_config['valve_moisture_threshold'] - moisture)
    temp_excess = max(0, temperature - crop_config['valve_temp_threshold'])
    
    # Determine if irrigation is needed
    needs_irrigation = (moisture < crop_config['valve_moisture_threshold']) or \
                       (temperature > crop_config['valve_temp_threshold'])
    
    if not needs_irrigation:
        return 0.0, 0.0, "None"
    
    # Calculate irrigation intensity (0-100 scale)
    intensity = 0
    if moisture < crop_config['valve_moisture_threshold']:
        intensity += (moisture_deficit / crop_config['valve_moisture_threshold']) * 50
    if temperature > crop_config['valve_temp_threshold']:
        intensity += (temp_excess / 10.0) * 50
    
    intensity = min(100, intensity)
    
    # Determine irrigation type based on intensity
    if intensity < 30:
        # Light irrigation - use efficient drip
        irrigation_type = "Drip"
        base_flow = IRRIGATION_FLOW_RATES['Drip']['base_flow']
        flow_std = IRRIGATION_FLOW_RATES['Drip']['std']
        duration_minutes = 20 + (intensity * 1.5)  # 20-65 minutes
    elif intensity < 60:
        # Moderate irrigation - use sprinkler
        irrigation_type = "Sprinkler"
        base_flow = IRRIGATION_FLOW_RATES['Sprinkler']['base_flow']
        flow_std = IRRIGATION_FLOW_RATES['Sprinkler']['std']
        duration_minutes = 15 + (intensity * 0.8)  # 15-63 minutes
    else:
        # Heavy irrigation - manual/hose for urgent watering
        irrigation_type = "Manual"
        base_flow = IRRIGATION_FLOW_RATES['Manual']['base_flow']
        flow_std = IRRIGATION_FLOW_RATES['Manual']['std']
        duration_minutes = 10 + (intensity * 0.6)  # 10-70 minutes
    
    # Add realistic variation to flow rate
    flow_rate = base_flow + np.random.normal(0, flow_std)
    flow_rate = max(0.5, flow_rate)  # Minimum 0.5 L/min
    
    # Add variation to duration
    duration_minutes += np.random.normal(0, 5)
    duration_minutes = max(0, duration_minutes)
    
    return round(flow_rate, 2), round(duration_minutes, 2), irrigation_type

def add_seasonal_variation(base_value, timestamp, is_temperature=False):
    """
    Add seasonal patterns to simulate real agricultural data
    Kenya has two rainy seasons: March-May (long rains) and October-December (short rains)
    """
    month = timestamp.month
    day_of_year = timestamp.timetuple().tm_yday
    
    # Seasonal multiplier based on Kenya's climate
    if is_temperature:
        # Temperature variations: hotter in Jan-Feb, cooler in June-Aug
        seasonal_factor = 2.0 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
        return base_value + seasonal_factor
    else:
        # Moisture variations: higher during rainy seasons
        if 3 <= month <= 5:  # Long rains (March-May)
            seasonal_factor = 8.0
        elif 10 <= month <= 12:  # Short rains (Oct-Dec)
            seasonal_factor = 5.0
        else:  # Dry seasons
            seasonal_factor = -3.0
        
        return base_value + seasonal_factor + np.random.normal(0, 2)

def add_diurnal_variation(base_value, hour, is_temperature=False):
    """
    Add daily (diurnal) patterns
    Temperature: cooler at night, hotter at midday
    Moisture: higher at night (less evaporation), lower during day
    """
    if is_temperature:
        # Peak temperature around 2 PM (14:00), coolest at 6 AM
        diurnal_factor = 4.0 * np.sin(2 * np.pi * (hour - 6) / 24)
        return base_value + diurnal_factor
    else:
        # Moisture decreases during day due to evapotranspiration
        diurnal_factor = -3.0 * np.sin(2 * np.pi * (hour - 6) / 24)
        return base_value + diurnal_factor

print("="*70)
print("GENERATING KENYA SMART IRRIGATION DATASET")
print("="*70)
print(f"\nFarm Location: {FARM_REGION}")
print(f"Total Records: {TOTAL_RECORDS}")
print(f"Date Range: {START_DATE.strftime('%Y-%m-%d')} onwards")
print(f"Crops: Maize (8,500 records), Tomatoes (8,500 records)")
print("\nIrrigation Flow Rates (based on industry standards):")
for irr_type, specs in IRRIGATION_FLOW_RATES.items():
    print(f"  {irr_type:12} - {specs['base_flow']:5.1f} L/min (±{specs['std']:.1f}) - {specs['description']}")
print("\nGenerating realistic sensor data with:")
print("  - Seasonal variations (rainy vs dry seasons)")
print("  - Diurnal patterns (day vs night)")
print("  - Irrigation valve flow rates (L/min) and duration (minutes)")
print("="*70)

# Generate dataset
records = []
current_time = START_DATE

for crop_name, crop_config in CROPS.items():
    print(f"\nGenerating {crop_config['records']} records for {crop_name}...")
    
    for i in range(crop_config['records']):
        # Generate timestamp (readings every ~30 minutes throughout the year)
        current_time += timedelta(minutes=np.random.randint(25, 35))
        
        # Extract time features
        hour = current_time.hour
        day_of_year = current_time.timetuple().tm_yday
        
        # Generate base moisture with seasonal and diurnal patterns
        base_moisture = crop_config['optimal_moisture']
        moisture = base_moisture + np.random.normal(0, crop_config['moisture_std'])
        moisture = add_seasonal_variation(moisture, current_time, is_temperature=False)
        moisture = add_diurnal_variation(moisture, hour, is_temperature=False)
        moisture = max(5.0, min(95.0, moisture))  # Realistic bounds
        
        # Generate base temperature with seasonal and diurnal patterns
        base_temp = crop_config['optimal_temp']
        temperature = base_temp + np.random.normal(0, crop_config['temp_std'])
        temperature = add_seasonal_variation(temperature, current_time, is_temperature=True)
        temperature = add_diurnal_variation(temperature, hour, is_temperature=True)
        temperature = max(10.0, min(42.0, temperature))  # Realistic bounds for Kenya
        
        # Calculate irrigation valve flow rate and duration
        flow_rate, duration, irrigation_type = calculate_valve_flow_and_time(
            moisture, temperature, crop_config
        )
        
        # Create record
        record = {
            'region': FARM_REGION,
            'crop_type': crop_name,
            'soil_moisture': round(moisture, 2),
            'soil_temperature': round(temperature, 2),
            'irrigation_type': irrigation_type,
            'timestamp': current_time.strftime('%d/%m/%Y'),
            'sensor_id': f"{SENSOR_ID_BASE}_{crop_name[:3].upper()}",
            'irrigation_valve_flow_rate': flow_rate,  # L/min
            'irrigation_valve_minutes': duration  # minutes
        }
        
        records.append(record)

# Create DataFrame
df = pd.DataFrame(records)

# Shuffle to mix crops (more realistic)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display statistics
print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)

print("\nOverall Statistics:")
print(f"Total Records: {len(df)}")
print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

print("\nSoil Moisture (%):")
print(f"  Mean: {df['soil_moisture'].mean():.2f}")
print(f"  Std: {df['soil_moisture'].std():.2f}")
print(f"  Min: {df['soil_moisture'].min():.2f}")
print(f"  Max: {df['soil_moisture'].max():.2f}")

print("\nSoil Temperature (°C):")
print(f"  Mean: {df['soil_temperature'].mean():.2f}")
print(f"  Std: {df['soil_temperature'].std():.2f}")
print(f"  Min: {df['soil_temperature'].min():.2f}")
print(f"  Max: {df['soil_temperature'].max():.2f}")

print("\nIrrigation Valve Flow Rate (L/min):")
print(f"  Mean: {df['irrigation_valve_flow_rate'].mean():.2f}")
print(f"  Std: {df['irrigation_valve_flow_rate'].std():.2f}")
print(f"  Min: {df['irrigation_valve_flow_rate'].min():.2f}")
print(f"  Max: {df['irrigation_valve_flow_rate'].max():.2f}")

print("\nIrrigation Duration (minutes):")
print(f"  Mean: {df['irrigation_valve_minutes'].mean():.2f}")
print(f"  Std: {df['irrigation_valve_minutes'].std():.2f}")
print(f"  Min: {df['irrigation_valve_minutes'].min():.2f}")
print(f"  Max: {df['irrigation_valve_minutes'].max():.2f}")

print(f"\nNo irrigation events: {(df['irrigation_valve_flow_rate'] == 0).sum()} records ({100*(df['irrigation_valve_flow_rate'] == 0).sum()/len(df):.1f}%)")

print("\nCrop Distribution:")
print(df['crop_type'].value_counts())

print("\nIrrigation Type Distribution:")
irrigation_dist = df['irrigation_type'].value_counts()
for irr_type, count in irrigation_dist.items():
    if irr_type != 'None':
        avg_flow = df[df['irrigation_type'] == irr_type]['irrigation_valve_flow_rate'].mean()
        print(f"  {irr_type:12} - {count:5d} records ({100*count/len(df):5.1f}%) | Avg flow: {avg_flow:5.2f} L/min")
    else:
        print(f"  {irr_type:12} - {count:5d} records ({100*count/len(df):5.1f}%)")

print("\n" + "="*70)
print("SAMPLE RECORDS")
print("="*70)
print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

print("\nRecords with high irrigation (flow > 30 L/min):")
high_irrigation = df[df['irrigation_valve_flow_rate'] > 30].head(5)
if len(high_irrigation) > 0:
    print(high_irrigation.to_string(index=False))
else:
    print("No records with flow > 30 L/min")

print("\nRecords with no irrigation:")
no_irrigation = df[df['irrigation_valve_flow_rate'] == 0].head(5)
print(no_irrigation.to_string(index=False))

# Save to CSV
output_file = 'soil_sensor_data.csv'
df.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ Dataset generated successfully!")
print(f"✓ Saved to: {output_file}")
print(f"✓ File size: {len(df)} records")
print("="*70)

# Display column info
print("\nColumn Information:")
print("-" * 70)
for col in df.columns:
    dtype = df[col].dtype
    unique = df[col].nunique()
    print(f"{col:35} | Type: {str(dtype):10} | Unique: {unique:6}")

print("\n" + "="*70)
print("IRRIGATION FLOW RATE REFERENCE")
print("="*70)
print("\nBased on agricultural irrigation standards:")
print("  Drip Irrigation:      1-4 L/min    (0.25-1 gph per emitter)")
print("  Sprinkler System:    10-25 L/min   (2.5-6.5 gpm for small farms)")
print("  Manual/Hose:         30-50 L/min   (8-13 gpm typical)")
print("\nConversions:")
print("  1 L/min = 0.264 gpm (gallons per minute)")
print("  1 gpm = 3.785 L/min")
print("="*70)

print("\n" + "="*70)
print("DATASET READY FOR MODEL TRAINING!")
print("="*70)
print("\nNext steps:")
print("1. Review the generated dataset: soil_sensor_data.csv")
print("2. Train the model: python train_model.py")
print("3. Start the backend: python app.py")
print("="*70)