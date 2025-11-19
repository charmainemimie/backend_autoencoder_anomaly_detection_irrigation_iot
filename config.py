"""
Configuration file for Smart Irrigation System
Adjust these parameters to tune anomaly detection sensitivity
"""

# ==================================================
# DATA SOURCE CONFIGURATION
# ==================================================

# Set to True when you have real IoT sensors connected
# Set to False for development/demo with simulated data
USE_REAL_SENSORS = False  # Default: False (simulation mode)

# If True, system will generate simulated data when no real data received for 5 minutes
# Useful for testing and preventing dashboard from freezing
FALLBACK_TO_SIMULATION = True  # Default: True

# ==================================================
# SENSOR SIMULATION SETTINGS
# ==================================================

# Base sensor values (mean values from your dataset)
BASE_MOISTURE = 42.0  # Percentage (%)
BASE_TEMPERATURE = 24.5  # Celsius (°C)

# Natural variation (standard deviation for random noise)
# Lower values = less variation = fewer false positives
MOISTURE_STD = 2.0  # Default: 2.0, Range: 1.0-5.0
TEMPERATURE_STD = 1.5  # Default: 1.5, Range: 0.5-3.0

# ==================================================
# ANOMALY DETECTION SETTINGS
# ==================================================

# Threshold multiplier (higher = less sensitive, fewer false positives)
# 1.0 = Use exact threshold from training (most sensitive)
# 1.5 = 50% more tolerant (recommended for production)
# 2.0 = 100% more tolerant (very relaxed)
THRESHOLD_MULTIPLIER = 1.5  # Default: 1.5

# ==================================================
# IRRIGATION VALVE LOGIC
# ==================================================

# Valve opens when moisture drops below this value OR temp exceeds max
MIN_MOISTURE_THRESHOLD = 30  # Percentage (%)
MAX_TEMPERATURE_THRESHOLD = 32  # Celsius (°C)

# ==================================================
# SYSTEM SETTINGS
# ==================================================

# Update frequency (seconds between readings)
UPDATE_INTERVAL = 1  # Default: 1 second

# Print anomaly alerts to console
PRINT_ANOMALY_ALERTS = True

# ==================================================
# TUNING GUIDE
# ==================================================

"""
IF YOU SEE TOO MANY FALSE ANOMALIES:
1. Increase THRESHOLD_MULTIPLIER to 2.0 or 2.5
2. Decrease MOISTURE_STD and TEMPERATURE_STD to 1.5 and 1.0
3. Retrain model if data distribution has changed

IF ATTACKS ARE NOT DETECTED:
1. Decrease THRESHOLD_MULTIPLIER to 1.0 or 1.2
2. Retrain model with more diverse data including attack patterns
3. Check if scaler.pkl matches your current data distribution

OPTIMAL SETTINGS (based on your dataset):
- THRESHOLD_MULTIPLIER: 1.3-1.8
- MOISTURE_STD: 1.5-2.5
- TEMPERATURE_STD: 1.0-2.0

TEST YOUR SETTINGS:
1. Start the backend
2. Watch for 1-2 minutes without attacks
3. Should see 0-2 anomalies per minute max
4. Then simulate attack - should detect within 2-5 seconds
"""