# day13_feature_extraction_v2.py
# Extracts 17 features (16 original + f17_dc_level) from every waveform.
# f17_dc_level = mean of last 20 points — directly separates RC_open (near 0V)
# from RC_short (near 5V), fixing the F1=0 failure identified in Day 16.

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

BASE    = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose"
CSV_IN  = os.path.join(BASE, "master_dataset_sweep.csv")
CSV_OUT = os.path.join(BASE, "features_v2.csv")
N_PTS   = 200

T_END = {
    "RC":         0.095,
    "RLC":        0.095,
    "RC_LADDER":  0.095,
    "SALLEN_KEY": 0.020
}

def f1_peak_voltage(v):        return float(np.max(v))
def f2_rms_value(v):           return float(np.sqrt(np.mean(v**2)))
def f3_mean_voltage(v):        return float(np.mean(v))
def f4_std_voltage(v):         return float(np.std(v))

def f5_rise_time(t, v):
    peak = np.max(v)
    if peak < 1e-6: return float(t[-1])
    idx_lo = np.argmax(v >= 0.1*peak)
    idx_hi = np.argmax(v >= 0.9*peak)
    if idx_hi <= idx_lo: return float(t[-1]-t[0])
    return float(t[idx_hi]-t[idx_lo])

def f6_settling_time(t, v):
    final = v[-1]
    if abs(final) < 1e-6: return float(t[-1])
    outside = np.where(np.abs(v-final) > 0.02*abs(final))[0]
    if len(outside)==0: return float(t[0])
    return float(t[outside[-1]])

def f7_skewness(v):            return float(skew(v))
def f8_kurtosis(v):            return float(kurtosis(v))

def f9_dominant_freq(v, t_end, n):
    dt=t_end/(n-1); freq=rfftfreq(n,d=dt); mag=np.abs(rfft(v))
    mask=(freq>=50)&(freq<=2000)
    if not np.any(mask): return float(freq[np.argmax(mag)])
    return float(freq[mask][np.argmax(mag[mask])])

def f10_spectral_energy(v, t_end, n):
    mag=np.abs(rfft(v)); return float(np.sum(mag**2))

def f11_spectral_centroid(v, t_end, n):
    dt=t_end/(n-1); freq=rfftfreq(n,d=dt); mag=np.abs(rfft(v))
    denom=np.sum(mag)
    if denom<1e-12: return 0.0
    return float(np.sum(freq*mag)/denom)

def f12_bandwidth(v, t_end, n):
    dt=t_end/(n-1); freq=rfftfreq(n,d=dt); mag=np.abs(rfft(v))
    denom=np.sum(mag)
    if denom<1e-12: return 0.0
    c=np.sum(freq*mag)/denom
    return float(np.sqrt(np.sum(((freq-c)**2)*mag)/denom))

def f13_spectral_entropy(v, t_end, n):
    mag=np.abs(rfft(v))**2; total=np.sum(mag)
    if total<1e-12: return 0.0
    p=mag/total; p=p[p>0]
    return float(-np.sum(p*np.log2(p)))

def f14_energy(v):             return float(np.sum(v**2))

def f15_zero_crossing_rate(v):
    return float(np.sum(np.diff(np.sign(v-np.mean(v)))!=0)/len(v))

def f16_crest_factor(v):
    rms=np.sqrt(np.mean(v**2))
    if rms<1e-12: return 0.0
    return float(np.max(np.abs(v))/rms)

def f17_dc_level(v):
    # Mean of last 20 points = steady-state DC level.
    # RC_open: near 0V (capacitor never charges in 95ms window)
    # RC_short: near 5V (capacitor charges instantly)
    # This directly separates the two previously confused classes.
    return float(np.mean(v[-20:]))

df_raw    = pd.read_csv(CSV_IN)
wave_cols = [f"v{i}" for i in range(N_PTS)]
records   = []

for idx, row in df_raw.iterrows():
    circ  = row["circuit"]
    t_end = T_END[circ]
    v     = row[wave_cols].values.astype(float)
    t     = np.linspace(0.0, t_end, N_PTS)
    rec = {
        "sample_id": int(row["sample_id"]),
        "circuit":   circ,
        "fault":     row["fault"],
        "f1":  f1_peak_voltage(v),
        "f2":  f2_rms_value(v),
        "f3":  f3_mean_voltage(v),
        "f4":  f4_std_voltage(v),
        "f5":  f5_rise_time(t,v),
        "f6":  f6_settling_time(t,v),
        "f7":  f7_skewness(v),
        "f8":  f8_kurtosis(v),
        "f9":  f9_dominant_freq(v,t_end,N_PTS),
        "f10": f10_spectral_energy(v,t_end,N_PTS),
        "f11": f11_spectral_centroid(v,t_end,N_PTS),
        "f12": f12_bandwidth(v,t_end,N_PTS),
        "f13": f13_spectral_entropy(v,t_end,N_PTS),
        "f14": f14_energy(v),
        "f15": f15_zero_crossing_rate(v),
        "f16": f16_crest_factor(v),
        "f17": f17_dc_level(v),
    }
    records.append(rec)
    if (idx+1)%100==0: print(f"  processed {idx+1}/{len(df_raw)}...")

df_feat = pd.DataFrame(records)
df_feat.to_csv(CSV_OUT, index=False)
print(f"\nSaved {len(df_feat)} rows -> {CSV_OUT}")
print(f"Columns: {len(df_feat.columns)} (sample_id + circuit + fault + f1-f17)")

# Quick diagnosis: RC_open vs RC_short on f17
rc_open  = df_feat[(df_feat["circuit"]=="RC")&(df_feat["fault"]=="open")]["f17"].mean()
rc_short = df_feat[(df_feat["circuit"]=="RC")&(df_feat["fault"]=="short")]["f17"].mean()
print(f"\nf17 diagnosis:")
print(f"  RC_open  mean dc_level = {rc_open:.4f} V")
print(f"  RC_short mean dc_level = {rc_short:.4f} V")
print(f"  Separation gap = {abs(rc_open-rc_short):.4f} V")
print(f"  Separable: {'YES' if abs(rc_open-rc_short) > 1.0 else 'MARGINAL'}")
