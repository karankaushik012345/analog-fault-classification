# day17_dashboard.py
# Generates a single-page results dashboard summarising
# the entire SN Bose project: dataset, features, SVM results.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

# ---- Config -------------------------------------------------------
BASE    = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose"
CSV_IN  = os.path.join(BASE, "features.csv")
OUT_DIR = os.path.join(BASE, "Day17_Dashboard")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS  = [f"f{i}" for i in range(1, 17)]
FEATURE_NAMES = [
    "peak_voltage", "rms_value", "mean_voltage", "std_voltage",
    "rise_time", "settling_time", "skewness", "kurtosis",
    "dominant_freq", "spectral_energy", "spec_centroid",
    "bandwidth", "spec_entropy", "energy",
    "zero_crossing_rate", "crest_factor"
]
RANDOM_STATE = 42

# ---- Load and prepare ---------------------------------------------
df = pd.read_csv(CSV_IN)
df["label"] = df["circuit"] + "_" + df["fault"]
le = LabelEncoder()
y  = le.fit_transform(df["label"])
X  = df[FEATURE_COLS].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
labels  = le.classes_

# ---- Train all models ---------------------------------------------
print("Training models...")
models = {
    "Linear\n(default)":  SVC(kernel="linear", C=1.0,
                               decision_function_shape="ovo",
                               random_state=RANDOM_STATE),
    "RBF\n(default)":     SVC(kernel="rbf", C=1.0, gamma="scale",
                               decision_function_shape="ovo",
                               random_state=RANDOM_STATE),
    "Poly\n(default)":    SVC(kernel="poly", C=1.0, gamma="scale",
                               degree=3, coef0=1,
                               decision_function_shape="ovo",
                               random_state=RANDOM_STATE),
    "Linear\n(tuned)":    SVC(kernel="linear", C=100,
                               decision_function_shape="ovo",
                               random_state=RANDOM_STATE),
    "RBF\n(tuned)":       SVC(kernel="rbf", C=100, gamma=0.1,
                               decision_function_shape="ovo",
                               random_state=RANDOM_STATE),
    "Poly\n(tuned)":      SVC(kernel="poly", C=100, gamma=0.1,
                               degree=3, coef0=1,
                               decision_function_shape="ovo",
                               random_state=RANDOM_STATE),
}

accs, f1s = {}, {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred      = clf.predict(X_test)
    accs[name]  = accuracy_score(y_test, y_pred) * 100
    f1s[name]   = f1_score(y_test, y_pred, average=None)
    print(f"  {name.replace(chr(10),' '):<22}: {accs[name]:.2f}%")

# Best model for confusion matrix
best_clf = models["RBF\n(tuned)"]
y_pred_best = best_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

# Linear model for feature importance
lin_clf = models["Linear\n(tuned)"]
importance = np.mean(np.abs(lin_clf.coef_), axis=0)
feat_imp_df = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "importance": importance
}).sort_values("importance", ascending=True)

# Per-circuit accuracy (best model)
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred_best)
circuit_accs  = {}
for circ in ["RC", "RLC", "SALLEN_KEY", "RC_LADDER"]:
    mask = np.array([circ + "_" in l for l in y_test_labels])
    if mask.sum() > 0:
        circuit_accs[circ] = (
            np.sum(y_test_labels[mask] == y_pred_labels[mask])
            / mask.sum() * 100
        )

# ---- Build dashboard ----------------------------------------------
print("\nBuilding dashboard...")
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('#f8f9fa')
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.45, wspace=0.35)

# Shared style
TITLE_KW  = dict(fontsize=13, fontweight='bold', pad=10)
COLORS_6  = ['#90caf9','#90caf9','#90caf9','#1565c0','#1565c0','#1565c0']

# ── Panel 1: Kernel accuracy comparison ----------------------------
ax1 = fig.add_subplot(gs[0, 0])
names = list(accs.keys())
vals  = list(accs.values())
bars  = ax1.bar(names, vals, color=COLORS_6, edgecolor='white',
                linewidth=0.8, width=0.6)
ax1.axhline(y=76.96, color='#d32f2f', linestyle='--',
            linewidth=1.5, label='Best (RBF tuned)')
ax1.set_ylim(60, 90)
ax1.set_ylabel("Test Accuracy (%)", fontsize=10)
ax1.set_title("Kernel Comparison\nDefault vs Tuned", **TITLE_KW)
ax1.legend(fontsize=8)
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
             fontweight='bold')
ax1.tick_params(axis='x', labelsize=8)
ax1.set_facecolor('#ffffff')

# ── Panel 2: Per-circuit accuracy ---------------------------------
ax2 = fig.add_subplot(gs[0, 1])
circ_names = list(circuit_accs.keys())
circ_vals  = list(circuit_accs.values())
circ_colors = ['#ef9a9a' if v < 70 else '#fff176' if v < 85
               else '#a5d6a7' for v in circ_vals]
bars2 = ax2.bar(circ_names, circ_vals, color=circ_colors,
                edgecolor='grey', linewidth=0.8, width=0.5)
ax2.set_ylim(0, 110)
ax2.set_ylabel("Accuracy (%)", fontsize=10)
ax2.set_title("Per-Circuit Accuracy\n(RBF Tuned)", **TITLE_KW)
for bar, val in zip(bars2, circ_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom',
             fontsize=9, fontweight='bold')
ax2.tick_params(axis='x', labelsize=9)
ax2.set_facecolor('#ffffff')

# ── Panel 3: Dataset composition ----------------------------------
ax3 = fig.add_subplot(gs[0, 2])
fault_counts = df.groupby("fault").size()
wedge_colors = ['#ef5350','#42a5f5','#66bb6a','#ffa726','#ab47bc']
wedges, texts, autotexts = ax3.pie(
    fault_counts.values,
    labels=fault_counts.index,
    autopct='%1.1f%%',
    colors=wedge_colors,
    startangle=90,
    textprops={'fontsize': 9}
)
ax3.set_title(f"Dataset Composition\n({len(df)} total samples)", **TITLE_KW)

# ── Panel 4: Feature importance ------------------------------------
ax4 = fig.add_subplot(gs[1, 0])
bar_colors = ['#1565c0' if v >= feat_imp_df['importance'].quantile(0.75)
              else '#90caf9' for v in feat_imp_df['importance']]
ax4.barh(feat_imp_df['feature'], feat_imp_df['importance'],
         color=bar_colors, edgecolor='white')
ax4.set_xlabel("Mean |Coefficient|", fontsize=10)
ax4.set_title("Feature Importance\n(Linear SVM Weights)", **TITLE_KW)
ax4.tick_params(axis='y', labelsize=8)
ax4.set_facecolor('#ffffff')

# ── Panel 5: Per-class F1 (best model) ----------------------------
ax5 = fig.add_subplot(gs[1, 1:])
f1_best  = f1_score(y_test, y_pred_best, average=None)
f1_df    = pd.DataFrame({"class": labels, "f1": f1_best})
f1_df    = f1_df.sort_values("f1", ascending=True)
f1_colors = ["#d32f2f" if f == 0 else "#ef9a9a" if f < 0.5
             else "#fff176" if f < 0.8 else "#a5d6a7"
             for f in f1_df["f1"]]
ax5.barh(f1_df["class"], f1_df["f1"], color=f1_colors,
         edgecolor='white', height=0.7)
ax5.axvline(x=0.5, color='orange', linestyle='--',
            alpha=0.7, linewidth=1.5)
ax5.axvline(x=0.8, color='green', linestyle='--',
            alpha=0.7, linewidth=1.5)
for i, (_, row) in enumerate(f1_df.iterrows()):
    ax5.text(row['f1'] + 0.01, i, f"{row['f1']:.2f}",
             va='center', fontsize=8)
ax5.set_xlabel("F1 Score", fontsize=10)
ax5.set_xlim(0, 1.2)
ax5.set_title("Per-Class F1 Score — RBF SVM (C=100, γ=0.1)", **TITLE_KW)
ax5.set_facecolor('#ffffff')

# ── Panel 6: Confusion matrix -------------------------------------
ax6 = fig.add_subplot(gs[2, :])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.3, ax=ax6, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 7})
ax6.set_xlabel("Predicted", fontsize=11)
ax6.set_ylabel("Actual", fontsize=11)
ax6.set_title(
    f"Confusion Matrix — RBF SVM (C=100, γ=0.1) | "
    f"Test Accuracy: {76.96:.2f}%",
    **TITLE_KW
)
ax6.tick_params(axis='x', labelsize=7, rotation=45)
ax6.tick_params(axis='y', labelsize=7, rotation=0)

# ── Main title -----------------------------------------------------
fig.suptitle(
    "Kernel-Based Fault Classification in Analog Circuits\n"
    "SN Bose National Centre — Results Dashboard",
    fontsize=16, fontweight='bold', y=0.98
)

# ── Save ----------------------------------------------------------
out_path = os.path.join(OUT_DIR, "results_dashboard.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"Dashboard saved -> {out_path}")
print("\nDay 17 complete.")