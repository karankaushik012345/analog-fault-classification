# day15_gridsearch.py
# GridSearchCV hyperparameter tuning for Linear, RBF, and Polynomial SVMs.
# Finds optimal C, gamma, degree for each kernel.
# Test set never touched during search — only used for final evaluation.

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ---- Config -------------------------------------------------------
BASE     = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose"
CSV_IN   = os.path.join(BASE, "features.csv")
OUT_FILE = os.path.join(BASE, "day15_results.txt")

FEATURE_COLS = [f"f{i}" for i in range(1, 17)]
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# Day 14 baseline for comparison
DAY14 = {"Linear": 76.47, "RBF": 70.59, "Polynomial": 74.02}

# ---- Search grids -------------------------------------------------
# Wide log-scale search for C and gamma
PARAM_GRIDS = {
    "Linear": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    },
    "RBF": {
        "C":     [0.1, 1, 10, 100, 1000],
        "gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10]
    },
    "Polynomial": {
        "C":      [0.1, 1, 10, 100],
        "gamma":  [0.001, 0.01, 0.1],
        "degree": [2, 3],
        "coef0":  [1]
    }
}

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

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Search grids:")
for k, g in PARAM_GRIDS.items():
    n = 1
    for v in g.values(): n *= len(v)
    print(f"  {k}: {n} combinations × {CV_FOLDS} folds = {n*CV_FOLDS} fits")
print()

# ---- Run GridSearchCV ---------------------------------------------
output_lines = []

def log(line=""):
    print(line)
    output_lines.append(line)

log("=" * 60)
log("DAY 15 — GRIDSEARCHCV HYPERPARAMETER TUNING")
log("=" * 60)
log(f"Dataset : 1019 samples | 16 features | 20 classes")
log(f"Split   : 80/20 stratified | CV folds: {CV_FOLDS}")
log()

tuned_results = {}

# Linear
log("─" * 60)
log("KERNEL: Linear")
log("─" * 60)
log(f"Search grid: C = {PARAM_GRIDS['Linear']['C']}")
gs_linear = GridSearchCV(
    SVC(kernel="linear", decision_function_shape="ovo",
        random_state=RANDOM_STATE),
    param_grid={"C": PARAM_GRIDS["Linear"]["C"]},
    cv=CV_FOLDS, scoring="accuracy", n_jobs=-1, verbose=1
)
gs_linear.fit(X_train, y_train)
best_linear = gs_linear.best_estimator_
y_pred      = best_linear.predict(X_test)
test_acc    = accuracy_score(y_test, y_pred)

log(f"Best params : C = {gs_linear.best_params_['C']}")
log(f"Best CV acc : {gs_linear.best_score_*100:.2f}%")
log(f"Test acc    : {test_acc*100:.2f}%  (Day14 default: {DAY14['Linear']}%)")
log(f"Improvement : {(test_acc*100 - DAY14['Linear']):+.2f}%")
log()
log("Classification Report:")
log(classification_report(y_test, y_pred,
    target_names=le.classes_, digits=3))
tuned_results["Linear"] = {
    "best_params": gs_linear.best_params_,
    "best_cv":     gs_linear.best_score_,
    "test_acc":    test_acc,
    "report":      classification_report(y_test, y_pred,
                   target_names=le.classes_, digits=3)
}

# RBF
log("─" * 60)
log("KERNEL: RBF")
log("─" * 60)
log(f"Search grid: C = {PARAM_GRIDS['RBF']['C']}, "
    f"gamma = {PARAM_GRIDS['RBF']['gamma']}")
gs_rbf = GridSearchCV(
    SVC(kernel="rbf", decision_function_shape="ovo",
        random_state=RANDOM_STATE),
    param_grid={"C": PARAM_GRIDS["RBF"]["C"],
                "gamma": PARAM_GRIDS["RBF"]["gamma"]},
    cv=CV_FOLDS, scoring="accuracy", n_jobs=-1, verbose=1
)
gs_rbf.fit(X_train, y_train)
best_rbf = gs_rbf.best_estimator_
y_pred   = best_rbf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

log(f"Best params : C = {gs_rbf.best_params_['C']}, "
    f"gamma = {gs_rbf.best_params_['gamma']}")
log(f"Best CV acc : {gs_rbf.best_score_*100:.2f}%")
log(f"Test acc    : {test_acc*100:.2f}%  (Day14 default: {DAY14['RBF']}%)")
log(f"Improvement : {(test_acc*100 - DAY14['RBF']):+.2f}%")
log()
log("Classification Report:")
log(classification_report(y_test, y_pred,
    target_names=le.classes_, digits=3))
tuned_results["RBF"] = {
    "best_params": gs_rbf.best_params_,
    "best_cv":     gs_rbf.best_score_,
    "test_acc":    test_acc,
    "report":      classification_report(y_test, y_pred,
                   target_names=le.classes_, digits=3)
}

# Polynomial
log("─" * 60)
log("KERNEL: Polynomial")
log("─" * 60)
log(f"Search grid: C={PARAM_GRIDS['Polynomial']['C']}, "
    f"gamma={PARAM_GRIDS['Polynomial']['gamma']}, "
    f"degree={PARAM_GRIDS['Polynomial']['degree']}")
gs_poly = GridSearchCV(
    SVC(kernel="poly", decision_function_shape="ovo",
        random_state=RANDOM_STATE),
    param_grid={"C":      PARAM_GRIDS["Polynomial"]["C"],
                "gamma":  PARAM_GRIDS["Polynomial"]["gamma"],
                "degree": PARAM_GRIDS["Polynomial"]["degree"],
                "coef0":  PARAM_GRIDS["Polynomial"]["coef0"]},
    cv=CV_FOLDS, scoring="accuracy", n_jobs=-1, verbose=1
)
gs_poly.fit(X_train, y_train)
best_poly = gs_poly.best_estimator_
y_pred    = best_poly.predict(X_test)
test_acc  = accuracy_score(y_test, y_pred)

log(f"Best params : C={gs_poly.best_params_['C']}, "
    f"gamma={gs_poly.best_params_['gamma']}, "
    f"degree={gs_poly.best_params_['degree']}, "
    f"coef0={gs_poly.best_params_['coef0']}")
log(f"Best CV acc : {gs_poly.best_score_*100:.2f}%")
log(f"Test acc    : {test_acc*100:.2f}%  (Day14 default: {DAY14['Polynomial']}%)")
log(f"Improvement : {(test_acc*100 - DAY14['Polynomial']):+.2f}%")
log()
log("Classification Report:")
log(classification_report(y_test, y_pred,
    target_names=le.classes_, digits=3))
tuned_results["Polynomial"] = {
    "best_params": gs_poly.best_params_,
    "best_cv":     gs_poly.best_score_,
    "test_acc":    test_acc,
    "report":      classification_report(y_test, y_pred,
                   target_names=le.classes_, digits=3)
}

# ---- Final summary ------------------------------------------------
log("=" * 60)
log("FINAL SUMMARY — DEFAULT vs TUNED")
log("=" * 60)
log(f"{'Kernel':<14} {'Default':>10} {'Tuned':>10} {'Gain':>10} {'Best Params'}")
log(f"{'─'*14} {'─'*10} {'─'*10} {'─'*10} {'─'*30}")
for name, r in tuned_results.items():
    gain = r['test_acc']*100 - DAY14[name]
    params_str = ", ".join(f"{k}={v}" for k,v in r['best_params'].items())
    log(f"{name:<14} {DAY14[name]:>9.2f}% "
        f"{r['test_acc']*100:>9.2f}% "
        f"{gain:>+9.2f}% "
        f"{params_str}")
log()

best_kernel = max(tuned_results, key=lambda k: tuned_results[k]['test_acc'])
log(f"Best overall kernel (tuned): {best_kernel} at "
    f"{tuned_results[best_kernel]['test_acc']*100:.2f}%")

# ---- Save ---------------------------------------------------------
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"\nResults saved -> {OUT_FILE}")
