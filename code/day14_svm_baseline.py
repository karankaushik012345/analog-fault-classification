# day14_svm_baseline.py
# Honest SVM baseline on real LTspice sweep features.
# Trains Linear, RBF, and Polynomial SVMs on features.csv
# and reports per-class and overall performance.

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

# ---- Config -------------------------------------------------------
BASE     = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose"
CSV_IN   = os.path.join(BASE, "features.csv")
OUT_FILE = os.path.join(BASE, "day14_results.txt")

FEATURE_COLS = [f"f{i}" for i in range(1, 17)]
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ---- Load data ----------------------------------------------------
df = pd.read_csv(CSV_IN)
print(f"Loaded {len(df)} samples, {len(FEATURE_COLS)} features")
print(f"Classes: {sorted(df['fault'].unique())}")
print(f"Circuits: {sorted(df['circuit'].unique())}\n")

# Create combined label: circuit_fault (20 classes)
df["label"] = df["circuit"] + "_" + df["fault"]
print(f"Total unique labels: {df['label'].nunique()}")
print(df.groupby("label").size().to_string())
print()

# ---- Encode and split ---------------------------------------------
le = LabelEncoder()
y  = le.fit_transform(df["label"])
X  = df[FEATURE_COLS].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

# ---- Scale --------------------------------------------------------
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ---- Define three kernels -----------------------------------------
kernels = {
    "Linear":     SVC(kernel="linear",  C=1.0,
                      random_state=RANDOM_STATE, decision_function_shape="ovo"),
    "RBF":        SVC(kernel="rbf",     C=1.0,  gamma="scale",
                      random_state=RANDOM_STATE, decision_function_shape="ovo"),
    "Polynomial": SVC(kernel="poly",    C=1.0,  gamma="scale", degree=3, coef0=1,
                      random_state=RANDOM_STATE, decision_function_shape="ovo"),
}

# ---- Train, evaluate, report --------------------------------------
results = {}
output_lines = []

def log(line=""):
    print(line)
    output_lines.append(line)

log("=" * 60)
log("DAY 14 —  SVM BASELINE ON REAL SWEPT DATA")
log("=" * 60)
log(f"Dataset : {len(df)} samples | {len(FEATURE_COLS)} features | 20 classes")
log(f"Split   : {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test | stratified")
log(f"Scaling : StandardScaler (zero mean, unit variance)")
log()

for name, clf in kernels.items():
    log(f"{'─'*60}")
    log(f"KERNEL: {name}")
    log(f"{'─'*60}")

    # Train
    clf.fit(X_train, y_train)

    # Test accuracy
    y_pred    = clf.predict(X_test)
    test_acc  = accuracy_score(y_test, y_pred)

    # 5-fold cross-validation on full dataset
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")

    log(f"Test  Accuracy : {test_acc*100:.2f}%")
    log(f"CV    Accuracy : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    log()

    # Per-class report
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=3
    )
    log("Classification Report:")
    log(report)
