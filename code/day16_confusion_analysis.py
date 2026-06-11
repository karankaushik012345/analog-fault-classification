# day16_confusion_analysis.py
# Full confusion matrix analysis and error diagnosis
# using the best model from Day 15: RBF C=100, gamma=0.1

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                    # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
import warnings
warnings.filterwarnings("ignore")

# ---- Config -------------------------------------------------------
BASE     = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose"
CSV_IN   = os.path.join(BASE, "features.csv")
OUT_DIR  = os.path.join(BASE, "Day16_Analysis")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [f"f{i}" for i in range(1, 17)]
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ---- Load and prepare ---------------------------------------------
df = pd.read_csv(CSV_IN)
df["label"] = df["circuit"] + "_" + df["fault"]

le = LabelEncoder()
y  = le.fit_transform(df["label"])
X  = df[FEATURE_COLS].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ---- Train best model (RBF tuned) ---------------------------------
print("Training best model: RBF C=100, gamma=0.1 ...")
clf = SVC(kernel="rbf", C=100, gamma=0.1,
          decision_function_shape="ovo", random_state=RANDOM_STATE)
clf.fit(X_train, y_train)
y_pred   = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_acc*100:.2f}%\n")

# ---- 1. Full 20x20 Confusion Matrix heatmap -----------------------
print("Generating confusion matrix heatmap...")
cm     = confusion_matrix(y_test, y_pred)
labels = le.classes_

fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix — RBF SVM (C=100, γ=0.1)\nTest Accuracy: {test_acc*100:.2f}%",
             fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
path1 = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path1}")

# ---- 2. Per-class F1 bar chart ------------------------------------
print("Generating per-class F1 chart...")
f1_scores = f1_score(y_test, y_pred, average=None)
f1_df     = pd.DataFrame({"class": labels, "f1": f1_scores})
f1_df     = f1_df.sort_values("f1", ascending=True)

colors = ["#d32f2f" if f < 0.5 else "#f57c00" if f < 0.8
          else "#388e3c" for f in f1_df["f1"]]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(f1_df["class"], f1_df["f1"], color=colors)
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='F1=0.5 threshold')
ax.axvline(x=0.8, color='orange', linestyle='--', alpha=0.5, label='F1=0.8 threshold')
for bar, val in zip(bars, f1_df["f1"]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel("F1 Score", fontsize=12)
ax.set_title("Per-Class F1 Score — RBF SVM (C=100, γ=0.1)", fontsize=14)
ax.set_xlim(0, 1.15)
ax.legend(fontsize=10)
plt.tight_layout()
path2 = os.path.join(OUT_DIR, "f1_per_class.png")
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path2}")

# ---- 3. Top confused pairs ----------------------------------------
print("\nTop 10 most confused class pairs:")
print(f"{'Actual':<25} {'Predicted':<25} {'Count':>6}")
print("-" * 58)
confused = []
for i in range(len(labels)):
    for j in range(len(labels)):
        if i != j and cm[i, j] > 0:
            confused.append((labels[i], labels[j], cm[i, j]))
confused.sort(key=lambda x: -x[2])
for actual, predicted, count in confused[:10]:
    print(f"{actual:<25} {predicted:<25} {count:>6}")

# ---- 4. Per-circuit accuracy breakdown ----------------------------
print("\nPer-circuit accuracy:")
print(f"{'Circuit':<15} {'Accuracy':>10} {'Correct':>8} {'Total':>8}")
print("-" * 45)

# Map test indices back to circuit labels
df_test = df.iloc[
    pd.Series(range(len(df))).sample(frac=TEST_SIZE,
    random_state=RANDOM_STATE).sort_values().index
]

circuits = ["RC", "RLC", "SALLEN_KEY", "RC_LADDER"]
y_test_labels   = le.inverse_transform(y_test)
y_pred_labels   = le.inverse_transform(y_pred)

for circ in circuits:
    mask    = np.array([circ + "_" in l for l in y_test_labels])
    if mask.sum() == 0:
        continue
    correct = np.sum(y_test_labels[mask] == y_pred_labels[mask])
    total   = mask.sum()
    acc     = correct / total * 100
    print(f"{circ:<15} {acc:>9.2f}% {correct:>8} {total:>8}")

# ---- 5. RC_open diagnosis -----------------------------------------
print("\n--- RC_open vs RC_short Feature Diagnosis ---")
rc_open  = df[(df["circuit"]=="RC") & (df["fault"]=="open")]
rc_short = df[(df["circuit"]=="RC") & (df["fault"]=="short")]

key_features = ["f1", "f2", "f3", "f4", "f5", "f14"]
feat_names   = {
    "f1": "peak_voltage", "f2": "rms_value",
    "f3": "mean_voltage", "f4": "std_voltage",
    "f5": "rise_time",    "f14": "energy"
}
print(f"\n{'Feature':<12} {'RC_open mean':>14} {'RC_short mean':>14} {'Separable?':>12}")
print("-" * 55)
for f in key_features:
    o_mean = rc_open[f].mean()
    s_mean = rc_short[f].mean()
    gap    = abs(o_mean - s_mean)
    sep    = "YES" if gap > 0.5 else "MARGINAL" if gap > 0.1 else "NO"
    print(f"{feat_names[f]:<12} {o_mean:>14.4f} {s_mean:>14.4f} {sep:>12}")

# ---- 6. Normal vs Drift diagnosis ---------------------------------
print("\n--- Normal vs Drift Feature Diagnosis (all circuits) ---")
normal = df[df["fault"]=="normal"]
drift  = df[df["fault"]=="drift"]
print(f"\n{'Feature':<12} {'Normal mean':>13} {'Drift mean':>13} {'Separable?':>12}")
print("-" * 52)
for f in ["f1","f2","f3","f5","f6","f9","f13"]:
    n_mean = normal[f].mean()
    d_mean = drift[f].mean()
    gap    = abs(n_mean - d_mean)
    sep    = "YES" if gap > 0.5 else "MARGINAL" if gap > 0.1 else "NO"
    print(f"{f:<12} {n_mean:>13.4f} {d_mean:>13.4f} {sep:>12}")

# ---- 7. Summary report --------------------------------------------
report = classification_report(y_test, y_pred,
                                target_names=labels, digits=3)
print(f"\nFull Classification Report:\n{report}")

summary = f"""
DAY 16 ANALYSIS SUMMARY
========================
Best Model    : RBF SVM, C=100, gamma=0.1
Test Accuracy : {test_acc*100:.2f}%

Perfect classes (F1=1.0):
{', '.join([labels[i] for i,f in enumerate(f1_scores) if f==1.0])}

Failed classes (F1=0):
{', '.join([labels[i] for i,f in enumerate(f1_scores) if f==0.0])}

Weak classes (F1 < 0.5):
{', '.join([labels[i] for i,f in enumerate(f1_scores) if 0 < f < 0.5])}
"""
print(summary)

out_txt = os.path.join(OUT_DIR, "day16_summary.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(summary)
    f.write("\nFull Classification Report:\n")
    f.write(report)
    f.write("\n\nTop Confused Pairs:\n")
    for actual, predicted, count in confused[:10]:
        f.write(f"{actual} -> {predicted}: {count}\n")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Files: confusion_matrix.png, f1_per_class.png, day16_summary.txt")