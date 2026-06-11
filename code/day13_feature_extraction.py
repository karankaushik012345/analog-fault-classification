# day13_feature_extraction.py
# Extracts 16 physical features from every waveform in
# master_dataset_sweep.csv and writes features.csv.
# Input:  ...\Final Sn Bose\master_dataset_sweep.csv
# Output: ...\Final Sn Bose\features.csv

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

# ---- Config -------------------------------------------------------
BASE    = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose"
CSV_IN  = os.path.join(BASE, "master_dataset_sweep.csv")
CSV_OUT = os.path.join(BASE, "features.csv")
N_PTS   = 200

# Time axis end per circuit (matches dt_map from Days 1-8)
T_END = {
    "RC":         0.095,
    "RLC":        0.095,
    "RC_LADDER":  0.095,
    "SALLEN_KEY": 0.020
}

# ---- Individual feature functions ---------------------------------

def f1_peak_voltage(v):
    return float(np.max(v))

def f2_rms_value(v):
    return float(np.sqrt(np.mean(v ** 2)))

def f3_mean_voltage(v):
    return float(np.mean(v))

def f4_std_voltage(v):
    return float(np.std(v))

def f5_rise_time(t, v):
    """Time from 10% to 90% of peak voltage."""
    peak = np.max(v)
    if peak < 1e-6:          # flat waveform (open fault) — rise time undefined
        return float(t[-1])
    lo = 0.1 * peak
    hi = 0.9 * peak
    idx_lo = np.argmax(v >= lo)
    idx_hi = np.argmax(v >= hi)
    if idx_hi <= idx_lo:
        return float(t[-1] - t[0])
    return float(t[idx_hi] - t[idx_lo])

def f6_settling_time(t, v):
    """Last time waveform is outside 2% band around final value."""
    final = v[-1]
    if abs(final) < 1e-6:    # flat waveform
        return float(t[-1])
    band = 0.02 * abs(final)
    outside = np.where(np.abs(v - final) > band)[0]
    if len(outside) == 0:
        return float(t[0])
    return float(t[outside[-1]])

def f7_skewness(v):
    return float(skew(v))

def f8_kurtosis(v):
    return float(kurtosis(v))

def f9_dominant_freq(v, t_end, n):
    """Dominant frequency searched in 50-2000 Hz only."""
    dt   = t_end / (n - 1)
    freq = rfftfreq(n, d=dt)
    mag  = np.abs(rfft(v))
    mask = (freq >= 50) & (freq <= 2000)
    if not np.any(mask):
        return float(freq[np.argmax(mag)])
    return float(freq[mask][np.argmax(mag[mask])])

def f10_spectral_energy(v, t_end, n):
    dt  = t_end / (n - 1)
    mag = np.abs(rfft(v))
    return float(np.sum(mag ** 2))

def f11_spectral_centroid(v, t_end, n):
    dt   = t_end / (n - 1)
    freq = rfftfreq(n, d=dt)
    mag  = np.abs(rfft(v))
    denom = np.sum(mag)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(freq * mag) / denom)

def f12_bandwidth(v, t_end, n):
    """Spectral bandwidth — std of frequency distribution."""
    dt      = t_end / (n - 1)
    freq    = rfftfreq(n, d=dt)
    mag     = np.abs(rfft(v))
    denom   = np.sum(mag)
    if denom < 1e-12:
        return 0.0
    centroid = np.sum(freq * mag) / denom
    return float(np.sqrt(np.sum(((freq - centroid) ** 2) * mag) / denom))

def f13_spectral_entropy(v, t_end, n):
    """Shannon entropy of normalised power spectrum."""
    mag   = np.abs(rfft(v)) ** 2
    total = np.sum(mag)
    if total < 1e-12:
        return 0.0
    p = mag / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def f14_energy(v):
    return float(np.sum(v ** 2))

def f15_zero_crossing_rate(v):
    """Number of zero crossings normalised by signal length."""
    crossings = np.sum(np.diff(np.sign(v - np.mean(v))) != 0)
    return float(crossings / len(v))

def f16_crest_factor(v):
    rms = np.sqrt(np.mean(v ** 2))
    if rms < 1e-12:
        return 0.0
    return float(np.max(np.abs(v)) / rms)

# ---- Main extraction loop -----------------------------------------
df_raw = pd.read_csv(CSV_IN)
wave_cols = [f"v{i}" for i in range(N_PTS)]

records = []
for idx, row in df_raw.iterrows():
    circ  = row["circuit"]
    fault = row["fault"]
    t_end = T_END[circ]
    v     = row[wave_cols].values.astype(float)
    t     = np.linspace(0.0, t_end, N_PTS)

    rec = {
        "sample_id": int(row["sample_id"]),
        "circuit":   circ,
        "fault":     fault,
        "f1":  f1_peak_voltage(v),
        "f2":  f2_rms_value(v),
        "f3":  f3_mean_voltage(v),
        "f4":  f4_std_voltage(v),
        "f5":  f5_rise_time(t, v),
        "f6":  f6_settling_time(t, v),
        "f7":  f7_skewness(v),
        "f8":  f8_kurtosis(v),
        "f9":  f9_dominant_freq(v, t_end, N_PTS),
        "f10": f10_spectral_energy(v, t_end, N_PTS),
        "f11": f11_spectral_centroid(v, t_end, N_PTS),
        "f12": f12_bandwidth(v, t_end, N_PTS),
        "f13": f13_spectral_entropy(v, t_end, N_PTS),
        "f14": f14_energy(v),
        "f15": f15_zero_crossing_rate(v),
        "f16": f16_crest_factor(v),
    }
    records.append(rec)

    if (idx + 1) % 100 == 0:
        print(f"  processed {idx+1}/{len(df_raw)} samples...")

df_feat = pd.DataFrame(records)
df_feat.to_csv(CSV_OUT, index=False)

print(f"\n{'='*50}")
print(f"Saved {len(df_feat)} rows -> {CSV_OUT}")
print(f"Columns: {list(df_feat.columns)}")
print(f"\nSamples per class:")
print(df_feat.groupby(["circuit","fault"]).size().to_string())
print(f"\nFeature statistics:")
print(df_feat[["f1","f2","f3","f4","f5","f6","f7","f8",
               "f9","f10","f11","f12","f13","f14","f15","f16"]].describe().round(4))