# Kernel-Based Fault Classification in Analog Circuits



[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://analogcircuitsfaultclassification.streamlit.app/)



Machine learning pipeline for classifying faults (Normal, Drift, Open, 

Short, Noise) across 4 analog circuit topologies (RC, RLC, Sallen-Key, 

RC Ladder) using LTspice simulation data and SVM kernel comparison.



## Live Demo

**[Try it here](https://analogcircuitsfaultclassification.streamlit.app/)**



## Overview

- 1019 samples generated from real LTspice `.step param` parameter sweep simulations

- 17 physically-derived features (time-domain + frequency-domain)

- SVM kernel comparison (Linear, RBF, Polynomial) with GridSearchCV tuning

- Topology-aware post-processing for cross-circuit fault disambiguation

- **Final accuracy: 78.43%** on 20-class fault classification problem



## Project Structure
fault-classification-project/
â”œâ”€â”€ app.py                  # Streamlit web application
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ master_dataset_sweep.csv   # Raw waveform dataset (1019 samples)
â”‚   â””â”€â”€ features_v2.csv            # Extracted 17-feature dataset
â”œâ”€â”€ model/
â”‚   â”œâ”€â”€ model.pkl            # Trained RBF SVM
â”‚   â”œâ”€â”€ scaler.pkl           # StandardScaler
â”‚   â””â”€â”€ label_encoder.pkl    # Label encoder for 20 classes
â”œâ”€â”€ code/                    # Full Python pipeline (sweep parsing,
â”‚                              feature extraction, training, dashboard)
â”œâ”€â”€ results/                 # Dashboard, classification reports
â””â”€â”€ ltspice_models/          # LTspice .asc files with sweep directives

## Methodology
1. **Circuit simulation** â€” 4 circuit topologies modeled in LTspice XVII:
   - RC low-pass filter
   - RLC series circuit (underdamped)
   - Sallen-Key active filter
   - RC ladder network (cascaded stages)

2. **Parameter sweeps** â€” `.step param` directive generates ~51 independent
   transient simulations per fault class by sweeping component values
   (resistance, capacitance) across physically meaningful ranges.

3. **Feature extraction** â€” 17 features per waveform:
   - Time-domain: peak voltage, RMS, rise time, settling time, skewness, kurtosis, dc_level
   - Frequency-domain: dominant frequency, spectral energy/centroid/entropy, bandwidth
   - Structural: zero-crossing rate, crest factor, energy

4. **Classification** â€” SVM with 3 kernels evaluated, hyperparameters tuned
   via GridSearchCV (5-fold CV), topology-aware post-processing applied.

## Results

| Kernel | Default | Tuned + Topology Rule |
|--------|---------|------------------------|
| Linear | 77.94%  | 77.45% |
| RBF    | 71.08%  | **78.43%** |
| Polynomial | 75.49% | 77.94% |

**Per-circuit accuracy:** RLC 100%, Sallen-Key 82.4%, RC 65.7%, RC-Ladder 60.0%

## Key Finding
Replacing Python-augmented training data with physics-based LTspice 
parameter sweeps reduced reported accuracy from 91% to ~76%, demonstrating 
that augmentation had inflated classifier performance by ~14 percentage 
points â€” the model was learning the augmentation transform rather than 
genuine circuit fault signatures.

## Tech Stack
Python Â· scikit-learn Â· LTspice XVII Â· pandas Â· matplotlib Â· scipy Â· Streamlit

## Author
Karan Kaushik â€” B.Tech ECE, NIT Silchar  
SN Bose Summer Internship, NIT Silchar â€” Internship 2026  
Supervisor: Dr. Anish Kumar Saha
