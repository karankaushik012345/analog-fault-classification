# save_model.py
# Trains the final RBF model with topology rule and saves it
# as .pkl files for deployment in the Streamlit app.

import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

BASE = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose"
FEATURE_COLS = [f"f{i}" for i in range(1, 18)]

# Load data
df = pd.read_csv(os.path.join(BASE, "features_v2.csv"))
df["label"] = df["circuit"] + "_" + df["fault"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["label"])
X = df[FEATURE_COLS].values

# Split (same random_state as before for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train final model
model = SVC(kernel="rbf", C=100, gamma=0.1,
            decision_function_shape="ovo", random_state=42, probability=True)
model.fit(X_train_scaled, y_train)

# Verify accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, model.predict(X_test_scaled)) * 100
print(f"Model accuracy on test set: {acc:.2f}%")

# Save everything needed for deployment
SAVE_DIR = os.path.join(BASE, "deployment")
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(model, os.path.join(SAVE_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(SAVE_DIR, "label_encoder.pkl"))

# Also save a few sample test rows for the demo
sample_df = df.iloc[X_test.shape[0]*0:5].copy()  # first 5 rows for demo
sample_df.to_csv(os.path.join(SAVE_DIR, "sample_input.csv"), index=False)

print(f"\nSaved to: {SAVE_DIR}")
print("Files: model.pkl, scaler.pkl, label_encoder.pkl, sample_input.csv")