import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal, norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, RocCurveDisplay)

df = pd.read_csv('celeba_baldvsnonbald_normalised.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Train anomalies: {y_train.sum()} ({y_train.mean():.2%})")
print(f"Test anomalies:  {y_test.sum()} ({y_test.mean():.2%})")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_normal_scaled = X_train_scaled[y_train.values == 0]
X_test_scaled  = scaler.transform(X_test)

print(f"Normal training instances: {X_train_normal_scaled.shape[0]}")

mu  = X_train_normal_scaled.mean(axis=0)
cov = np.cov(X_train_normal_scaled, rowvar=False)

print(f"Mean vector shape:       {mu.shape}")
print(f"Covariance matrix shape: {cov.shape}")
print(f"Covariance matrix rank:  {np.linalg.matrix_rank(cov)}")

try:
    _ = np.linalg.inv(cov)
    print("Covariance matrix is invertible: YES — full covariance model usable")
except np.linalg.LinAlgError:
    print("Covariance matrix is singular — use diagonal covariance instead")

rv = multivariate_normal(mean=mu, cov=cov, allow_singular=False)

log_probs = rv.logpdf(X_test_scaled)

anomaly_scores = -log_probs

print(f"Score range:        {anomaly_scores.min():.4f} to {anomaly_scores.max():.4f}")
print(f"Normal mean score:  {anomaly_scores[y_test == 0].mean():.4f}")
print(f"Anomaly mean score: {anomaly_scores[y_test == 1].mean():.4f}")
print(f"Separation ratio:   {anomaly_scores[y_test==1].mean() / anomaly_scores[y_test==0].mean():.2f}x")

precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, anomaly_scores)
f1_vals = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-9)
best_idx       = f1_vals.argmax()
best_threshold = thresholds[best_idx]

y_pred = (anomaly_scores >= best_threshold).astype(int)

print(f"Best threshold: {best_threshold:.4f}")
print(f"Precision: {precision_vals[best_idx]:.4f} | "f"Recall: {recall_vals[best_idx]:.4f} | "f"F1: {f1_vals[best_idx]:.4f}")

print("=" * 55)
print("GAUSSIAN PARAMETRIC MODEL — RESULTS")
print("=" * 55)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)
print(f"\nTrue Positives  (anomalies correctly flagged): {tp}")
print(f"False Positives (normal incorrectly flagged):  {fp}")
print(f"True Negatives  (normal correctly cleared):    {tn}")
print(f"False Negatives (anomalies missed):            {fn}")

avg_prec = average_precision_score(y_test, anomaly_scores)
print(f"ROC-AUC:        {roc_auc_score(y_test, anomaly_scores):.4f}")
print(f"Average Precision: {avg_prec:.4f}")

means_diag = X_train_normal_scaled.mean(axis=0)
stds_diag  = X_train_normal_scaled.std(axis=0)
scores_diag = -np.sum(norm.logpdf(X_test_scaled, loc=means_diag, scale=stds_diag), axis=1)

print(f"Diagonal ROC-AUC: {roc_auc_score(y_test, scores_diag):.4f}")

print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"Avg Precision: {average_precision_score(y_test, anomaly_scores):.4f}")
print(f"Best threshold: {best_threshold:.4f}")

# ── 8. Diagonal comparison ────────────────────────────────────────────────────
means_diag = X_train_normal_scaled.mean(axis=0)
stds_diag  = X_train_normal_scaled.std(axis=0)
scores_diag = -np.sum(norm.logpdf(X_test_scaled,
                                   loc=means_diag,
                                   scale=stds_diag), axis=1)
print(f"Diagonal ROC-AUC: {roc_auc_score(y_test, scores_diag):.4f}")

# ── 9. Plots ──────────────────────────────────────────────────────────────────
# Confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix — Gaussian Parametric Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('gaussian_confusion_matrix.png', dpi=150)
plt.show()

# Score distribution
plt.figure(figsize=(9, 5))
plot_df = pd.DataFrame({'score': anomaly_scores, 'label': y_test.values})
sns.histplot(data=plot_df, x='score', hue='label', bins=80,
             stat='density', common_norm=False,
             palette={0: 'steelblue', 1: 'tomato'}, alpha=0.6)
plt.axvline(best_threshold, color='black', linestyle='--',
            label=f'Threshold = {best_threshold:.2f}')
plt.title('Negative Log-Likelihood — Normal vs Anomaly')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('gaussian_score_distribution.png', dpi=150)
plt.show()

# PR curve
plt.figure(figsize=(7, 5))
plt.plot(recall_vals, precision_vals, color='darkorange', lw=2,
         label=f'PR Curve (AP={average_precision_score(y_test, anomaly_scores):.3f})')
plt.scatter(recall_vals[best_idx], precision_vals[best_idx],
            color='red', zorder=5,
            label=f'Best threshold (F1={f1_vals[best_idx]:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — Gaussian Parametric Model')
plt.legend()
plt.tight_layout()
plt.savefig('gaussian_pr_curve.png', dpi=150)
plt.show()