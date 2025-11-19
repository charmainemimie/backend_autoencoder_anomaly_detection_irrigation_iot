import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import json

# Load your dataset
print("Loading dataset...")
df = pd.read_csv('soil_sensor_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Data preprocessing
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Extract features for anomaly detection
# Using soil_moisture and soil_temperature as main features
X = df[['soil_moisture', 'soil_temperature']].values

# Remove any rows with NaN values
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
print(f"\nClean data samples: {X.shape[0]}")

# Basic statistics
print(f"\nSoil Moisture - Mean: {X[:, 0].mean():.2f}, Std: {X[:, 0].std():.2f}, Range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
print(f"Soil Temperature - Mean: {X[:, 1].mean():.2f}, Std: {X[:, 1].std():.2f}, Range: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")

# Split data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {X_train_scaled.shape[0]}")
print(f"Testing samples: {X_test_scaled.shape[0]}")

# Build Autoencoder
print("\n" + "="*50)
print("BUILDING AUTOENCODER MODEL")
print("="*50)

input_dim = X_train_scaled.shape[1]
encoding_dim = 4

encoder = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.1),
    layers.Dense(encoding_dim, activation='relu')
], name='encoder')

decoder = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(encoding_dim,)),
    layers.Dropout(0.1),
    layers.Dense(input_dim, activation='linear')
], name='decoder')

autoencoder = keras.Sequential([encoder, decoder], name='autoencoder')

autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nModel Architecture:")
autoencoder.summary()

# Train model
print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate model
print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)

train_loss, train_mae = autoencoder.evaluate(X_train_scaled, X_train_scaled, verbose=0)
test_loss, test_mae = autoencoder.evaluate(X_test_scaled, X_test_scaled, verbose=0)

print(f"\nTraining Loss (MSE): {train_loss:.6f}")
print(f"Training MAE: {train_mae:.6f}")
print(f"Testing Loss (MSE): {test_loss:.6f}")
print(f"Testing MAE: {test_mae:.6f}")

# Calculate reconstruction errors for threshold
X_train_pred = autoencoder.predict(X_train_scaled, verbose=0)
train_mse = np.mean(np.power(X_train_scaled - X_train_pred, 2), axis=1)

# Set threshold at 95th percentile
threshold = np.percentile(train_mse, 95)
print(f"\nAnomaly Detection Threshold (95th percentile): {threshold:.6f}")

# Statistics on training reconstruction errors
print(f"Reconstruction Error - Mean: {train_mse.mean():.6f}, Std: {train_mse.std():.6f}")
print(f"Reconstruction Error - Min: {train_mse.min():.6f}, Max: {train_mse.max():.6f}")

# Test on test set
X_test_pred = autoencoder.predict(X_test_scaled, verbose=0)
test_mse = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)
anomalies_detected = np.sum(test_mse > threshold)

print(f"\nTest Set Analysis:")
print(f"Anomalies detected: {anomalies_detected}/{len(test_mse)} ({100*anomalies_detected/len(test_mse):.2f}%)")

# Distribution of reconstruction errors in test set
print(f"Test Reconstruction Error - Mean: {test_mse.mean():.6f}, Std: {test_mse.std():.6f}")
print(f"Test Reconstruction Error - Min: {test_mse.min():.6f}, Max: {test_mse.max():.6f}")

# Save model and scaler
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

autoencoder.save('autoencoder_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Calculate additional metrics
print("\n" + "="*50)
print("CALCULATING CLASSIFICATION METRICS")
print("="*50)

# For anomaly detection evaluation, treat top 5% as "true anomalies"
y_true_simulated = (test_mse > np.percentile(test_mse, 95)).astype(int)
y_pred_test = (test_mse > threshold).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true_simulated, y_pred_test)
precision = precision_score(y_true_simulated, y_pred_test, zero_division=0)
recall = recall_score(y_true_simulated, y_pred_test, zero_division=0)
f1 = f1_score(y_true_simulated, y_pred_test, zero_division=0)

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score: {f1:.4f}")

print("\nNote: These metrics are calculated using simulated labels where")
print("the top 5% highest reconstruction errors are treated as anomalies.")
print("Run 'evaluate_model.py' for comprehensive evaluation metrics.")

# Save comprehensive metrics
metrics = {
    'dataset_info': {
        'total_samples': int(X.shape[0]),
        'training_samples': int(X_train_scaled.shape[0]),
        'testing_samples': int(X_test_scaled.shape[0])
    },
    'feature_statistics': {
        'moisture_mean': float(X[:, 0].mean()),
        'moisture_std': float(X[:, 0].std()),
        'temperature_mean': float(X[:, 1].mean()),
        'temperature_std': float(X[:, 1].std())
    },
    'model_performance': {
        'train_loss': float(train_loss),
        'train_mae': float(train_mae),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae)
    },
    'classification_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    'anomaly_detection': {
        'threshold': float(threshold),
        'test_anomalies_count': int(anomalies_detected),
        'test_anomalies_percentage': float(100*anomalies_detected/len(test_mse))
    }
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Model training complete!")
print("✓ Files saved:")
print("  - autoencoder_model.h5")
print("  - scaler.pkl")
print("  - model_metrics.json")
print("\n✓ Next step: Update app.py with the load_model.py code and run the Flask server")