from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Analog Circuit Fault Classifier",
    layout="wide"
)

# ---- Header ----
st.title("Kernel-Based Fault Classification in Analog Circuits")
st.markdown("""
**SVM-based fault classifier** trained on real LTspice parameter sweep simulation data.
Classifies faults across **4 circuit topologies** (RC, RLC, Sallen-Key, RC Ladder)
and **5 fault types** (Normal, Drift, Open, Short, Noise).

**Final accuracy: 78.43%** on 20-class problem (RBF kernel + topology-aware correction)
""")

# ---- Load model artifacts ----
@st.cache_resource
def load_artifacts():
    model  = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le     = joblib.load("model/label_encoder.pkl")
    return model, scaler, le

model, scaler, le = load_artifacts()

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["Dashboard", "Try Prediction", "Dataset"])

# ===== TAB 1: Dashboard =====
with tab1:
    st.header("Results Dashboard")
    if os.path.exists("results/results_dashboard_FINAL.png"):
        st.image(str(BASE_DIR / "results" / "results_dashboard_FINAL.png"), use_container_width=True)
    else:
        st.warning("Dashboard image not found")

    st.header("Final Results Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", "78.43%")
    col2.metric("RLC Accuracy", "100%")
    col3.metric("RC_open F1 (recovered)", "0.462")

    st.markdown("""
    | Kernel | Default | Tuned + Topology Rule |
    |--------|---------|------------------------|
    | Linear | 77.94% | 77.45% |
    | RBF    | 71.08% | **78.43%** |
    | Polynomial | 75.49% | 77.94% |
    """)

# ===== TAB 2: Try Prediction =====
with tab2:
    st.header("Try a Prediction")
    st.write("""
    Select a sample from the dataset or enter feature values manually 
    to see the model's fault classification.
    """)

    df = pd.read_csv("data/features_v2.csv")
    df["label"] = df["circuit"] + "_" + df["fault"]

    feature_cols = [f"f{i}" for i in range(1, 18)]
    feature_names = [
        "peak_voltage","rms_value","mean_voltage","std_voltage",
        "rise_time","settling_time","skewness","kurtosis",
        "dominant_freq","spectral_energy","spec_centroid",
        "bandwidth","spec_entropy","energy",
        "zero_crossing_rate","crest_factor","dc_level"
            format_func=lambda i: f"#{i} — {df.loc[i,'label']}"

    mode = st.radio("Input method:", ["Pick a sample from dataset", "Enter values manually"])

    if mode == "Pick a sample from dataset":
        sample_idx = st.selectbox(
            "Choose a sample (shown as circuit_fault, true label):",
            options=df.index,
            format_func=lambda i: f"#{i} ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â {df.loc[i,'label']}"
        )
        row = df.loc[sample_idx]
        true_label = row["label"]
        X_input = row[feature_cols].values.reshape(1, -1)
        circuit_input = row["circuit"]

        st.write(f"**True label:** {true_label}")

    else:
        st.write("Enter the 17 feature values:")
        cols = st.columns(3)
        values = []
        for i, fname in enumerate(feature_names):
            with cols[i % 3]:
                val = st.number_input(f"{fname} (f{i+1})", value=0.0, format="%.4f")
                values.append(val)
        X_input = np.array(values).reshape(1, -1)
        circuit_input = st.selectbox("Circuit type (for topology rule):",
                                     ["RC", "RLC", "SALLEN_KEY", "RC_LADDER"])
        true_label = None

    if st.button("Predict Fault Type", type="primary"):
        X_scaled = scaler.transform(X_input)
            st.info("Topology-aware correction applied: RLC_open → RC_open "
        pred_label = le.inverse_transform([pred])[0]

        # Apply topology-aware rule
        labels_list = list(le.classes_)
        rlc_open_idx = labels_list.index("RLC_open")
        rc_open_idx  = labels_list.index("RC_open")

                st.error(f"Incorrect — true label was {true_label}")
            pred = rc_open_idx
            pred_label = "RC_open"
            st.info("Topology-aware correction applied: RLC_open ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ RC_open "
                   "(an RC circuit cannot have an RLC-type fault)")

        st.success(f"**Predicted:** {pred_label}")

        if true_label is not None:
            if pred_label == true_label:
                st.success("Correct prediction!")
            else:
                st.error(f"Incorrect ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â true label was {true_label}")

        # Show prediction probabilities
        probs = model.predict_proba(X_scaled)[0]
        prob_df = pd.DataFrame({
            "Class": le.classes_,
            "Probability": probs
        }).sort_values("Probability", ascending=False).head(5)
        st.write("**Top 5 class probabilities:**")
        st.dataframe(prob_df, hide_index=True)

# ===== TAB 3: Dataset =====
with tab3:
    st.header("Dataset Overview")
    df = pd.read_csv("data/features_v2.csv")
    df["label"] = df["circuit"] + "_" + df["fault"]

    st.write(f"**Total samples:** {len(df)}")
    st.write(f"**Classes:** {df['label'].nunique()} (4 circuits â€” 5 fault types)")

    st.subheader("Samples per class")
    counts = df.groupby(["circuit", "fault"]).size().reset_index(name="count")
    st.dataframe(counts, hide_index=True)

    st.subheader("Feature columns (f1-f17)")
    feat_table = pd.DataFrame({
        "Feature": [f"f{i}" for i in range(1,18)],
        "Name": [
            "peak_voltage","rms_value","mean_voltage","std_voltage",
            "rise_time","settling_time","skewness","kurtosis",
            "dominant_freq","spectral_energy","spec_centroid",
            "bandwidth","spec_entropy","energy",
            "zero_crossing_rate","crest_factor","dc_level"
        ]
    })
    st.dataframe(feat_table, hide_index=True)

    st.subheader("Raw data sample")
    st.dataframe(df.head(10))

# ---- Footer ----
st.markdown("---")
st.markdown("""
**Project:** SN Bose Summer Internship 2026, NIT Silchar  
**Methodology:** LTspice parameter sweeps â†’ 17-feature extraction â†’ SVM kernel comparison â†’ GridSearchCV tuning â†’ topology-aware post-processing
""")
