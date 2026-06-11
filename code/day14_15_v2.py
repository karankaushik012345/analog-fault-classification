# day14_15_v2.py
# SVM baseline + GridSearchCV tuning on 17 features (f1-f17).
# Uses n_jobs=1 to avoid joblib hanging on this machine.

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE         = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose"
CSV_IN       = os.path.join(BASE, "features_v2.csv")
OUT_FILE     = os.path.join(BASE, "day14_15_v2_results.txt")
FEATURE_COLS = [f"f{i}" for i in range(1, 18)]   # 17 features
RANDOM_STATE = 42

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

output_lines = []
def log(line=""):
    print(line)
    output_lines.append(line)

log("="*60)
log("OPTION B — 17 FEATURES (added f17_dc_level)")
log("="*60)
log(f"Features: {len(FEATURE_COLS)} | Samples: {len(df)} | Classes: 20")
log()

# ── Baseline (default params) ─────────────────────────────────
log("--- BASELINE (default params) ---")
for name, kern, params in [
    ("Linear",     "linear", {"C":1.0}),
    ("RBF",        "rbf",    {"C":1.0, "gamma":"scale"}),
    ("Polynomial", "poly",   {"C":1.0, "gamma":"scale","degree":3,"coef0":1}),
]:
    clf = SVC(kernel=kern, decision_function_shape="ovo",
              random_state=RANDOM_STATE, **params)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))*100
    log(f"  {name:<12}: {acc:.2f}%")

log()

# ── GridSearchCV tuning (n_jobs=1 to avoid hanging) ───────────
log("--- GRIDSEARCHCV TUNING (n_jobs=1) ---")
log()

# Linear
log("KERNEL: Linear")
gs_lin = GridSearchCV(
    SVC(kernel="linear", decision_function_shape="ovo", random_state=RANDOM_STATE),
    {"C":[0.01,0.1,1,10,100,1000]},
    cv=5, scoring="accuracy", n_jobs=1, verbose=0
)
gs_lin.fit(X_train, y_train)
y_pred   = gs_lin.best_estimator_.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)*100
log(f"  Best C={gs_lin.best_params_['C']} | CV={gs_lin.best_score_*100:.2f}% | Test={test_acc:.2f}%")
log()

# RBF
log("KERNEL: RBF")
gs_rbf = GridSearchCV(
    SVC(kernel="rbf", decision_function_shape="ovo", random_state=RANDOM_STATE),
    {"C":[1,10,100,1000], "gamma":[0.001,0.01,0.1,1]},
    cv=5, scoring="accuracy", n_jobs=1, verbose=0
)
gs_rbf.fit(X_train, y_train)
y_pred_rbf = gs_rbf.best_estimator_.predict(X_test)
test_acc_rbf = accuracy_score(y_test, y_pred_rbf)*100
log(f"  Best C={gs_rbf.best_params_['C']}, gamma={gs_rbf.best_params_['gamma']}")
log(f"  CV={gs_rbf.best_score_*100:.2f}% | Test={test_acc_rbf:.2f}%")
log()

# Polynomial
log("KERNEL: Polynomial")
gs_poly = GridSearchCV(
    SVC(kernel="poly", decision_function_shape="ovo", random_state=RANDOM_STATE),
    {"C":[1,10,100], "gamma":[0.01,0.1], "degree":[2,3], "coef0":[1]},
    cv=5, scoring="accuracy", n_jobs=1, verbose=0
)
gs_poly.fit(X_train, y_train)
y_pred_poly = gs_poly.best_estimator_.predict(X_test)
test_acc_poly = accuracy_score(y_test, y_pred_poly)*100
log(f"  Best C={gs_poly.best_params_['C']}, gamma={gs_poly.best_params_['gamma']}, degree={gs_poly.best_params_['degree']}")
log(f"  CV={gs_poly.best_score_*100:.2f}% | Test={test_acc_poly:.2f}%")
log()

# ── Best model full report ─────────────────────────────────────
best_acc   = max(test_acc, test_acc_rbf, test_acc_poly)
if best_acc == test_acc_rbf:
    best_clf   = gs_rbf.best_estimator_
    best_pred  = y_pred_rbf
    best_name  = "RBF"
elif best_acc == test_acc:
    best_clf   = gs_lin.best_estimator_
    best_pred  = y_pred
    best_name  = "Linear"
else:
    best_clf   = gs_poly.best_estimator_
    best_pred  = y_pred_poly
    best_name  = "Polynomial"

log("="*60)
log(f"BEST MODEL: {best_name} at {best_acc:.2f}%")
log("="*60)
log()
log("Classification Report:")
log(classification_report(y_test, best_pred, target_names=labels, digits=3))

# RC_open specifically
f1_scores  = f1_score(y_test, best_pred, average=None)
rc_open_idx = list(labels).index("RC_open")
log(f"RC_open F1 (v1 = 0.000): {f1_scores[rc_open_idx]:.3f}")
log()

# Improvement summary
log("--- IMPROVEMENT vs 16-FEATURE VERSION ---")
log(f"  16 features best: 76.96% (RBF, C=100, gamma=0.1)")
log(f"  17 features best: {best_acc:.2f}% ({best_name})")
log(f"  Overall gain    : {best_acc-76.96:+.2f}%")
log(f"  RC_open F1 gain : 0.000 -> {f1_scores[rc_open_idx]:.3f}")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"\nResults saved -> {OUT_FILE}")
