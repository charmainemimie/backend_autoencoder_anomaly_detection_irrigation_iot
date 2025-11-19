import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, accuracy_score
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
import json

print("="*70)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*70)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('soil_sensor_data.csv')
X = df[['soil_moisture', 'soil_temperature']].values

# Remove NaN
mask = ~np.isnan(X).any(axis=1)
X = X[mask]

# Load trained model and scaler
print("Loading trained model and scaler...")
autoencoder = keras.models.load_model('autoencoder_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Scale data
X_scaled = scaler.transform(X)

# Get predictions
print("Generating predictions...")
X_reconstructed = autoencoder.predict(X_scaled, verbose=0)

# Calculate reconstruction errors
reconstruction_errors = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)

# Load threshold
with open('model_metrics.json', 'r') as f:
    metrics = json.load(f)
threshold = metrics['anomaly_detection']['threshold']

print(f"\nAnomaly Detection Threshold: {threshold:.6f}")

# Predict anomalies
y_pred = (reconstruction_errors > threshold).astype(int)

print("\n" + "="*70)
print("RECONSTRUCTION PERFORMANCE")
print("="*70)

# Reconstruction metrics
mse = np.mean(reconstruction_errors)
mae = np.mean(np.abs(X_scaled - X_reconstructed))
rmse = np.sqrt(mse)

print(f"\nMean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Per-feature reconstruction error
feature_names = ['soil_moisture', 'soil_temperature']
for i, feature in enumerate(feature_names):
    feature_mse = np.mean(np.power(X_scaled[:, i] - X_reconstructed[:, i], 2))
    feature_mae = np.mean(np.abs(X_scaled[:, i] - X_reconstructed[:, i]))
    print(f"\n{feature}:")
    print(f"  MSE: {feature_mse:.6f}")
    print(f"  MAE: {feature_mae:.6f}")

print("\n" + "="*70)
print("ANOMALY DETECTION STATISTICS")
print("="*70)

# Anomaly statistics
n_anomalies = np.sum(y_pred)
anomaly_rate = 100 * n_anomalies / len(y_pred)

print(f"\nTotal Samples: {len(y_pred)}")
print(f"Normal Samples: {len(y_pred) - n_anomalies} ({100 - anomaly_rate:.2f}%)")
print(f"Anomalies Detected: {n_anomalies} ({anomaly_rate:.2f}%)")

# Reconstruction error statistics
print(f"\nReconstruction Error Statistics:")
print(f"  Mean: {reconstruction_errors.mean():.6f}")
print(f"  Std: {reconstruction_errors.std():.6f}")
print(f"  Min: {reconstruction_errors.min():.6f}")
print(f"  Max: {reconstruction_errors.max():.6f}")
print(f"  Median: {np.median(reconstruction_errors):.6f}")
print(f"  25th Percentile: {np.percentile(reconstruction_errors, 25):.6f}")
print(f"  75th Percentile: {np.percentile(reconstruction_errors, 75):.6f}")
print(f"  95th Percentile (Threshold): {threshold:.6f}")
print(f"  99th Percentile: {np.percentile(reconstruction_errors, 99):.6f}")

# Simulated ground truth for demonstration
# In production, you would have labeled data with known anomalies
print("\n" + "="*70)
print("CLASSIFICATION METRICS (Simulated Validation)")
print("="*70)
print("\nNote: These metrics are based on the assumption that the top 5%")
print("reconstruction errors represent true anomalies for validation purposes.")
print("In production, compare against actual labeled attack/anomaly data.")

# Create simulated labels (top 5% as anomalies)
y_true_simulated = (reconstruction_errors > np.percentile(reconstruction_errors, 95)).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(y_true_simulated, y_pred)
precision = precision_score(y_true_simulated, y_pred, zero_division=0)
recall = recall_score(y_true_simulated, y_pred, zero_division=0)
f1 = f1_score(y_true_simulated, y_pred, zero_division=0)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true_simulated, y_pred)
print(f"\nConfusion Matrix:")
print(f"                 Predicted Normal  Predicted Anomaly")
print(f"Actual Normal         {cm[0,0]:6d}            {cm[0,1]:6d}")
print(f"Actual Anomaly        {cm[1,0]:6d}            {cm[1,1]:6d}")

# Calculate specificity and sensitivity
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nSensitivity (True Positive Rate): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"Specificity (True Negative Rate): {specificity:.4f} ({specificity*100:.2f}%)")
print(f"False Positive Rate: {fp/(fp+tn):.4f} ({100*fp/(fp+tn):.2f}%)")
print(f"False Negative Rate: {fn/(fn+tp):.4f} ({100*fn/(fn+tp):.2f}%)")

# Detailed classification report
print(f"\nDetailed Classification Report:")
print(classification_report(y_true_simulated, y_pred, 
                          target_names=['Normal', 'Anomaly'],
                          zero_division=0))

# ROC-AUC Score
try:
    roc_auc = roc_auc_score(y_true_simulated, reconstruction_errors)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
except:
    print("ROC-AUC Score: Unable to calculate")

# Precision-Recall AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_true_simulated, reconstruction_errors)
pr_auc = auc(recall_curve, precision_curve)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

print("\n" + "="*70)
print("ATTACK DETECTION CAPABILITY")
print("="*70)

# Simulate different attack scenarios
print("\nSimulating various attack intensities...")

attack_scenarios = [
    {'name': 'Mild Attack (+10 moisture)', 'moisture_delta': 10, 'temp_delta': 0},
    {'name': 'Moderate Attack (+20 moisture)', 'moisture_delta': 20, 'temp_delta': 0},
    {'name': 'Severe Attack (+30 moisture)', 'moisture_delta': 30, 'temp_delta': 0},
    {'name': 'Temperature Attack (+10°C)', 'moisture_delta': 0, 'temp_delta': 10},
    {'name': 'Combined Attack (+15 both)', 'moisture_delta': 15, 'temp_delta': 15},
]

print(f"\n{'Attack Scenario':<35} {'Detection Rate':<20} {'Avg Error':<15}")
print("-" * 70)

for scenario in attack_scenarios:
    # Create attacked samples
    X_attacked = X.copy()
    X_attacked[:100, 0] += scenario['moisture_delta']  # Attack moisture
    X_attacked[:100, 1] += scenario['temp_delta']       # Attack temperature
    
    X_attacked_scaled = scaler.transform(X_attacked)
    X_attacked_reconstructed = autoencoder.predict(X_attacked_scaled[:100], verbose=0)
    
    errors = np.mean(np.power(X_attacked_scaled[:100] - X_attacked_reconstructed, 2), axis=1)
    detection_rate = np.sum(errors > threshold) / len(errors)
    avg_error = np.mean(errors)
    
    print(f"{scenario['name']:<35} {detection_rate*100:>6.2f}%              {avg_error:>8.6f}")

print("\n" + "="*70)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("="*70)

# Analyze different threshold values
thresholds_to_test = [
    np.percentile(reconstruction_errors, 90),
    np.percentile(reconstruction_errors, 95),
    np.percentile(reconstruction_errors, 97),
    np.percentile(reconstruction_errors, 99)
]

print(f"\n{'Percentile':<15} {'Threshold':<15} {'Anomaly Rate':<15} {'F1-Score':<15}")
print("-" * 70)

for pct, thresh in zip([90, 95, 97, 99], thresholds_to_test):
    y_pred_thresh = (reconstruction_errors > thresh).astype(int)
    anomaly_rate_thresh = 100 * np.sum(y_pred_thresh) / len(y_pred_thresh)
    f1_thresh = f1_score(y_true_simulated, y_pred_thresh, zero_division=0)
    
    print(f"{pct}th%            {thresh:<15.6f} {anomaly_rate_thresh:<15.2f} {f1_thresh:<15.4f}")

# Save comprehensive metrics
comprehensive_metrics = {
    'reconstruction_performance': {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae)
    },
    'anomaly_statistics': {
        'total_samples': int(len(y_pred)),
        'normal_samples': int(len(y_pred) - n_anomalies),
        'anomalies_detected': int(n_anomalies),
        'anomaly_rate_percent': float(anomaly_rate)
    },
    'classification_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc) if 'roc_auc' in locals() else None,
        'pr_auc': float(pr_auc)
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'reconstruction_error_stats': {
        'mean': float(reconstruction_errors.mean()),
        'std': float(reconstruction_errors.std()),
        'min': float(reconstruction_errors.min()),
        'max': float(reconstruction_errors.max()),
        'median': float(np.median(reconstruction_errors)),
        'percentile_25': float(np.percentile(reconstruction_errors, 25)),
        'percentile_75': float(np.percentile(reconstruction_errors, 75)),
        'percentile_95': float(threshold),
        'percentile_99': float(np.percentile(reconstruction_errors, 99))
    }
}

with open('comprehensive_metrics.json', 'w') as f:
    json.dump(comprehensive_metrics, f, indent=2)

print("\n" + "="*70)
print("✓ Comprehensive evaluation complete!")
print("✓ Metrics saved to: comprehensive_metrics.json")
print("="*70)