# day12_build_sweep_dataset.py
# Rebuilds the master dataset from REAL LTspice parameter sweeps.

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---- Config -------------------------------------------------------
BASE   = r"C:\Users\karan\Desktop\May-June_july\Final Sn Bose\Sweeps"
T_END  = {"RC": 0.095, "RLC": 0.095, "RC_LADDER": 0.095, "SALLEN_KEY": 0.020}
N_PTS  = 200
CIRCS  = ["RC", "RLC", "SALLEN_KEY", "RC_LADDER"]
FAULTS = ["normal", "drift", "open", "short"]

# ---- File finder --------------------------------------------------
def find_sweep_file(folder, circuit, fault):
    stems = [
        f"{circuit.lower()}_{fault}_sweep",
        f"{circuit}_{fault}_sweep",
        f"Sallen_key_{fault}_sweep",
        f"sallen_key_{fault}_sweep",
    ]
    exts = [".txt", ""]
    for s in stems:
        for e in exts:
            p = os.path.join(folder, s + e)
            if os.path.isfile(p) and os.path.getsize(p) > 10_000:
                return p
    return None

# ---- Parser -------------------------------------------------------
def parse_step_file(fpath):
    runs, cur_t, cur_v, cur_p, prev_t = [], [], [], None, None

    def flush():
        nonlocal cur_t, cur_v, cur_p
        if len(cur_t) > 5:
            runs.append((cur_p, np.array(cur_t, dtype=float),
                                np.array(cur_v, dtype=float)))
        cur_t, cur_v = [], []

    with open(fpath, "r", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            lo = line.lower()
            if lo.startswith("step information"):
                flush()
                try:
                    cur_p = float(line.split("=")[1].split()[0].rstrip(",)"))
                except Exception:
                    cur_p = np.nan
                prev_t = None
                continue
            if lo.startswith("time") or lo.startswith("step"):
                continue
            parts = line.replace("\t", " ").replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                t, v = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if prev_t is not None and t < prev_t - 1e-12:
                flush()
                cur_p = None
            cur_t.append(t); cur_v.append(v); prev_t = t

    flush()
    return runs

# ---- Resample -----------------------------------------------------
def resample(t, v, t_end, n=N_PTS):
    grid = np.linspace(0.0, t_end, n)
    grid = np.clip(grid, t.min(), t.max())
    return interp1d(t, v, kind="linear",
                    bounds_error=False,
                    fill_value=(v[0], v[-1]))(grid)

# ---- Build dataset ------------------------------------------------
rows, sid = [], 0
for circ in CIRCS:
    cdir = os.path.join(BASE, circ)
    for fault in FAULTS:
        fpath = find_sweep_file(cdir, circ, fault)
        if fpath is None:
            print(f"[MISSING] {circ}/{fault} — no file found in {cdir}")
            continue
        print(f"[reading] {circ}/{fault}: {os.path.basename(fpath)}"
              f"  ({os.path.getsize(fpath)//1024} KB)")
        runs = parse_step_file(fpath)
        print(f"          -> {len(runs)} runs parsed")
        for pval, t, v in runs:
            wave = resample(t, v, T_END[circ])
            row = {"sample_id": sid, "circuit": circ,
                   "fault": fault, "param_value": pval}
            row.update({f"v{i}": wave[i] for i in range(N_PTS)})
            rows.append(row); sid += 1

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(BASE), "master_dataset_sweep.csv")
df.to_csv(out, index=False)
print(f"\n{'='*50}")
print(f"Saved {len(df)} samples  ->  {out}")
print(f"Columns: {len(df.columns)}")
print(f"\nSamples per class:")
print(df.groupby(["circuit", "fault"]).size().to_string())