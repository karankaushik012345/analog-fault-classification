# day12b_generate_noise_samples.py
# Generates noise-fault samples per circuit by injecting
# Gaussian noise onto the NORMAL sweep waveforms.
# Noise is physically justified: real circuit noise is stochastic,
# not a deterministic component shift.

import os
import numpy as np
import pandas as pd

# ---- Config -------------------------------------------------------
BASE     = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose"
CSV_IN   = os.path.join(BASE, "master_dataset_sweep.csv")
SNR_DB   = 25
SEED     = 42
CIRCUITS = ["RC", "RLC", "SALLEN_KEY", "RC_LADDER"]
N_PTS    = 200

rng = np.random.default_rng(SEED)

# ---- Load existing dataset ----------------------------------------
df        = pd.read_csv(CSV_IN)
wave_cols = [f"v{i}" for i in range(N_PTS)]
next_id   = df["sample_id"].max() + 1

noise_rows = []

for circ in CIRCUITS:
    normals      = df[(df["circuit"] == circ) & (df["fault"] == "normal")]
    n_noise      = len(normals)          # match exactly — 51 per circuit
    normal_waves = normals[wave_cols].values

    for i in range(n_noise):
        base        = normal_waves[i % len(normal_waves)]
        sig_power   = np.mean(base ** 2)
        noise_power = sig_power / (10 ** (SNR_DB / 10))
        noise_std   = np.sqrt(noise_power)
        noisy       = base + rng.normal(0, noise_std, size=N_PTS)

        row = {
            "sample_id":   next_id,
            "circuit":     circ,
            "fault":       "noise",
            "param_value": SNR_DB
        }
        row.update({f"v{i}": noisy[i] for i in range(N_PTS)})
        noise_rows.append(row)
        next_id += 1

    print(f"{circ}/noise: {n_noise} samples generated at {SNR_DB} dB SNR")

# ---- Append and save ----------------------------------------------
df_noise  = pd.DataFrame(noise_rows)
df_final  = pd.concat([df, df_noise], ignore_index=True)

out = os.path.join(BASE, "master_dataset_sweep.csv")
df_final.to_csv(out, index=False)

print(f"\nFinal dataset: {len(df_final)} samples")
print(df_final.groupby(["circuit", "fault"]).size().to_string())