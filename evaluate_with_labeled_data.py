"""
Evaluate trained autoencoder on test dataset with LABELED anomalies
This gives TRUE performance metrics using real anomaly labels
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, auc
)
from tensorflow import keras
import pickle
import json
import matplotlib.pyplot as plt

print("="*70)
print("EVALUATION WITH LABELED ANOMALIES")
print("="*70)

# Load test dataset with labels
print("\n📂 Loading labeled test dataset...")
try:
    df_test = pd.read_csv('test_dataset_with_attacks.csv')
    print(f"✓ Loaded: {len(df_test)} samples")
except FileNotFoundError:
    print("❌ Error: test_dataset_with_attacks.csv not found!")
    print("   Run: python generate_test_dataset_with_anomalies.py first")
    exit(1)

# Extract features and labels
X_test = df_test[['soil_moisture', 'soil_temperature']].values
y_true = df_test['is_anomaly'].values
anomaly_types = df_test['anomaly_type'].values

print(f"\n📊 Test Set Composition:")
print(f"  Total samples:   {len(y_true)}")
print(f"  Normal (0):      {np.sum(y_true==0)} ({100*np.sum(y_true==0)/len(y_true):.1f}%)")
print(f"  Anomalous (1):   {np.sum(y_true==1)} ({100*np.sum(y_true==1)/len(y_true):.1f}%)")

# Load trained model
print("\n🔄 Loading trained model...")
try:
    autoencoder = keras.models.load_model('autoencoder_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    threshold = metrics['anomaly_detection']['threshold']
    print(f"✓ Model loaded successfully")
    print(f"✓ Threshold: {threshold:.6f}")
except FileNotFoundError:
    print("❌ Error: Model files not found!")
    print("   Run: python train_model.py first")
    exit(1)

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Get predictions
print("\n🔄 Generating predictions...")
X_test_reconstructed = autoencoder.predict(X_test_scaled, verbose=0)

# Calculate reconstruction errors
reconstruction_errors = np.mean(np.power(X_test_scaled - X_test_reconstructed, 2), axis=1)

# Make predictions using threshold
y_pred = (reconstruction_errors > threshold).astype(int)

print("\n" + "="*70)
print("CLASSIFICATION METRICS (TRUE LABELS)")
print("="*70)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\n📊 Overall Performance:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Confusion Matrix:")
print(f"                 Predicted Normal  Predicted Anomaly")
print(f"Actual Normal         {tn:6d}            {fp:6d}")
print(f"Actual Anomaly        {fn:6d}            {tp:6d}")

# Detailed metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\n📊 Detection Rates:")
print(f"  Sensitivity (TPR):    {sensitivity:.4f} ({sensitivity*100:.2f}%) - Detects {sensitivity*100:.1f}% of anomalies")
print(f"  Specificity (TNR):    {specificity:.4f} ({specificity*100:.2f}%) - Correctly IDs {specificity*100:.1f}% of normal")
print(f"  False Positive Rate:  {fpr:.4f} ({fpr*100:.2f}%) - {fpr*100:.1f}% false alarms")
print(f"  False Negative Rate:  {fnr:.4f} ({fnr*100:.2f}%) - Misses {fnr*100:.1f}% of anomalies")

# Detailed classification report
print(f"\n📊 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], zero_division=0))

# ROC-AUC
try:
    roc_auc = roc_auc_score(y_true, reconstruction_errors)
    print(f"📊 ROC-AUC Score: {roc_auc:.4f}")
except:
    print("⚠ Could not calculate ROC-AUC")

# Precision-Recall AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_true, reconstruction_errors)
pr_auc = auc(recall_curve, precision_curve)
print(f"📊 Precision-Recall AUC: {pr_auc:.4f}")

# Per-anomaly-type analysis
print("\n" + "="*70)
print("PER-ANOMALY-TYPE DETECTION RATES")
print("="*70)

print(f"\n{'Anomaly Type':<25} {'Count':>8} {'Detected':>8} {'Rate':>8} {'Avg Error':>12}")
print("-"*70)

for anomaly_type in np.unique(anomaly_types[y_true == 1]):
    mask = (anomaly_types == anomaly_type) & (y_true == 1)
    count = np.sum(mask)
    detected = np.sum(y_pred[mask] == 1)
    rate = detected / count if count > 0 else 0
    avg_error = np.mean(reconstruction_errors[mask])
    
    print(f"{anomaly_type:<25} {count:>8d} {detected:>8d} {rate*100:>7.1f}% {avg_error:>12.6f}")

# Error distribution analysis
print("\n" + "="*70)
print("RECONSTRUCTION ERROR ANALYSIS")
print("="*70)

print(f"\n📊 Normal Samples (y_true=0):")
normal_errors = reconstruction_errors[y_true == 0]
print(f"  Mean:   {normal_errors.mean():.6f}")
print(f"  Median: {np.median(normal_errors):.6f}")
print(f"  Std:    {normal_errors.std():.6f}")
print(f"  Min:    {normal_errors.min():.6f}")
print(f"  Max:    {normal_errors.max():.6f}")
print(f"  95th percentile: {np.percentile(normal_errors, 95):.6f}")

print(f"\n📊 Anomalous Samples (y_true=1):")
anomaly_errors = reconstruction_errors[y_true == 1]
print(f"  Mean:   {anomaly_errors.mean():.6f}")
print(f"  Median: {np.median(anomaly_errors):.6f}")
print(f"  Std:    {anomaly_errors.std():.6f}")
print(f"  Min:    {anomaly_errors.min():.6f}")
print(f"  Max:    {anomaly_errors.max():.6f}")
print(f"  5th percentile: {np.percentile(anomaly_errors, 5):.6f}")

print(f"\n📊 Separation Analysis:")
separation_ratio = anomaly_errors.mean() / normal_errors.mean()
print(f"  Anomaly/Normal error ratio: {separation_ratio:.2f}x")
print(f"  Threshold: {threshold:.6f}")
print(f"  Anomalies above threshold: {np.sum(anomaly_errors > threshold)} / {len(anomaly_errors)}")
print(f"  Normals below threshold: {np.sum(normal_errors <= threshold)} / {len(normal_errors)}")

# Threshold sensitivity analysis
print("\n" + "="*70)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("="*70)

thresholds_to_test = [
    np.percentile(normal_errors, 90),
    np.percentile(normal_errors, 95),
    np.percentile(normal_errors, 97),
    np.percentile(normal_errors, 99),
    threshold  # Current threshold
]

print(f"\n{'Threshold':>12} {'Percentile':>12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*70)

for thresh in sorted(set(thresholds_to_test)):
    y_pred_thresh = (reconstruction_errors > thresh).astype(int)
    acc = accuracy_score(y_true, y_pred_thresh)
    prec = precision_score(y_true, y_pred_thresh, zero_division=0)
    rec = recall_score(y_true, y_pred_thresh, zero_division=0)
    f1_thresh = f1_score(y_true, y_pred_thresh, zero_division=0)
    
    # Find percentile
    pct = np.searchsorted(np.sort(normal_errors), thresh) / len(normal_errors) * 100
    
    marker = " ← CURRENT" if abs(thresh - threshold) < 1e-6 else ""
    print(f"{thresh:>12.6f} {pct:>11.1f}% {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1_thresh:>10.4f}{marker}")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if recall < 0.80:
    print("\n⚠ Low Recall ({:.1f}%): Missing many anomalies".format(recall*100))
    print("  → Consider lowering threshold (decrease THRESHOLD_MULTIPLIER)")
    better_thresh = np.percentile(normal_errors, 90)
    print(f"  → Try threshold: {better_thresh:.6f} (90th percentile)")

if fpr > 0.10:
    print("\n⚠ High False Positive Rate ({:.1f}%): Too many false alarms".format(fpr*100))
    print("  → Consider raising threshold (increase THRESHOLD_MULTIPLIER)")
    better_thresh = np.percentile(normal_errors, 97)
    print(f"  → Try threshold: {better_thresh:.6f} (97th percentile)")

if accuracy > 0.95 and recall > 0.80 and fpr < 0.05:
    print("\n✅ EXCELLENT PERFORMANCE!")
    print("  • High accuracy (>95%)")
    print("  • Good recall (>80%)")
    print("  • Low false positives (<5%)")
    print("  • Model is production-ready")

# Save results
results = {
    'test_set_info': {
        'total_samples': int(len(y_true)),
        'normal_samples': int(np.sum(y_true == 0)),
        'anomalous_samples': int(np.sum(y_true == 1)),
        'anomaly_types': df_test[df_test['is_anomaly']==1]['anomaly_type'].value_counts().to_dict()
    },
    'overall_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'sensitivity_tpr': float(sensitivity),
        'specificity_tnr': float(specificity),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'roc_auc': float(roc_auc) if 'roc_auc' in locals() else None,
        'pr_auc': float(pr_auc)
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'error_statistics': {
        'normal_mean': float(normal_errors.mean()),
        'normal_std': float(normal_errors.std()),
        'anomaly_mean': float(anomaly_errors.mean()),
        'anomaly_std': float(anomaly_errors.std()),
        'separation_ratio': float(separation_ratio)
    },
    'threshold_info': {
        'current_threshold': float(threshold),
        'normal_95th_percentile': float(np.percentile(normal_errors, 95)),
        'anomalies_detected': int(np.sum(y_pred == 1)),
        'anomalies_missed': int(fn)
    }
}

# Per-anomaly-type results
per_type_results = {}
for anomaly_type in np.unique(anomaly_types[y_true == 1]):
    mask = (anomaly_types == anomaly_type) & (y_true == 1)
    count = int(np.sum(mask))
    detected = int(np.sum(y_pred[mask] == 1))
    per_type_results[anomaly_type] = {
        'count': count,
        'detected': detected,
        'detection_rate': float(detected / count if count > 0 else 0),
        'avg_error': float(np.mean(reconstruction_errors[mask]))
    }
results['per_anomaly_type'] = per_type_results

output_file = 'evaluation_results_labeled.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("✓ EVALUATION COMPLETE!")
print("="*70)
print(f"\n✓ Results saved to: {output_file}")
print(f"\n📊 Key Metrics:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1:.4f}")
print("\n" + "="*70)