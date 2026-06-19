#!/usr/bin/env python3

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

bmi_cfe = None

# -----------------------
# Config
# -----------------------

DEFAULT_OBS_DIR      = "/mnt/disk2/1400_sites_helene/catchment_ts_03463300_dynamic_variance_rekrig"
DEFAULT_PARAMS_DIR   = "/mnt/disk2/1400_sites_helene/da_results_dynamic_novrugt_seeded"
DEFAULT_CFE_DIR      = "/mnt/disk2/suma_helen_poster/cfe_py"
DEFAULT_CONFIG_FILE  = "/mnt/disk2/suma_helen_poster/run_gpu/cat_03463300_bmi_config_cfe.json"
DEFAULT_FORCING_DIR1 = "/mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/output_03463300_nwmoperational/03463300/2023_2024_feb/forcings"
DEFAULT_FORCING_DIR2 = "/mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/output_03463300_nwmoperational/03463300/2024_feb_2025_sep/forcings"
DEFAULT_OUT_DIR      = "/mnt/disk2/1400_sites_helene/da_results_enkf"

SPINUP_START = "2023-02-01 00:00:00"
SPINUP_END   = "2023-09-30 23:00:00"
TEST_START   = "2023-10-01 00:00:00"
TEST_END     = "2024-10-31 23:00:00"

# -----------------------
# Ensemble spread controls
# -----------------------

PRECIP_SIGMA = 0.50
PRECIP_BIAS_SIGMA = 0.40
PET_SIGMA = 0.20

INIT_STATE_SIGMA = 0.20
SOIL_NOISE = 0.010
GW_NOISE   = 0.008
NASH_NOISE = 0.020

R_FLOOR = 1e-6

# -----------------------
# PET
# -----------------------

def pet(srad, T):
    T = T - 273.15
    delta = 4098 * (0.6108 * np.exp(17.27 * T / (T + 237.3))) / (T + 237.3) ** 2
    gamma = 0.0638
    Rn = np.maximum(0.8 * srad, 0.0) * 0.0036
    lam = 2.501 - 0.002361 * T
    return np.maximum(1.26 * (delta / (delta + gamma)) * Rn / lam, 0.0)

# -----------------------
# IO
# -----------------------

def load_forcing(p):
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["P"] = df["APCP_surface"] * 3600.0
    df["PET"] = pet(df["DSWRF_surface"].values, df["TMP_2maboveground"].values)
    return df


def load_obs(p):
    df = pd.read_csv(p)
    d = dict(zip(pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
                 df["qkrig_mm_hr"]))
    v = dict(zip(pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
                 df["qkrig_variance"]))
    return d, v


def combine_forcing(cat, d1, d2, outdir):
    out = outdir / f"{cat}_forcing.csv"
    if out.exists():
        return out
    df = pd.concat([pd.read_csv(Path(d1)/f"{cat}.csv"),
                    pd.read_csv(Path(d2)/f"{cat}.csv")])
    df = df.drop_duplicates("time").sort_values("time")
    df.to_csv(out, index=False)
    return out

# -----------------------
# Ensemble init
# -----------------------

def build_ensemble(n, cfg, forcing, rng):
    forcing = forcing.rename(columns={"date": "time"})

    bias = rng.lognormal(-0.5 * PRECIP_BIAS_SIGMA**2,
                         PRECIP_BIAS_SIGMA, n)

    models = []
    for i in range(n):
        m = bmi_cfe.BMI_CFE(cfg_file=cfg)
        m.load_forcing_file = lambda self=m: setattr(self, "forcing_data", forcing)
        m.initialize()

        if i > 0:
            m.soil_reservoir["storage_m"] *= (1 + INIT_STATE_SIGMA*rng.standard_normal())
            m.gw_reservoir["storage_m"]   *= (1 + INIT_STATE_SIGMA*rng.standard_normal())

        models.append(m)

    return models, bias

# -----------------------
# Step
# -----------------------

def step(models, bias, P, PET, rng):
    n = len(models)

    noise = rng.lognormal(-0.5*PRECIP_SIGMA**2, PRECIP_SIGMA, n)
    Pm = P * bias * noise
    PETm = np.maximum(PET * (1 + PET_SIGMA*rng.standard_normal(n)), 0)

    Q = np.zeros(n)

    for i, m in enumerate(models):
        m.set_value("atmosphere_water__time_integral_of_precipitation_mass_flux",
                    Pm[i]/1000)
        m.set_value("water_potential_evaporation_flux",
                    PETm[i]/1000/3600)
        m.update()
        Q[i] = m.get_value("land_surface_water__runoff_depth") * 1000

    return Q


def process_noise(models, rng):
    for m in models:
        m.soil_reservoir["storage_m"] *= (1 + SOIL_NOISE*rng.standard_normal())
        m.gw_reservoir["storage_m"]   *= (1 + GW_NOISE*rng.standard_normal())
        m.nash_storage += NASH_NOISE*rng.standard_normal(2)

# -----------------------
# EnKF
# -----------------------

def enkf(models, Q, y, R, rng):
    n = len(models)
    if np.any(np.isnan(Q)): return False

    R = max(R, R_FLOOR)

    sm = np.array([m.get_value("SOIL_CONCEPTUAL_STORAGE") for m in models])
    gw = np.array([m.gw_reservoir["storage_m"] for m in models])

    q = Q - Q.mean()
    Pyy = (q@q)/(n-1)
    Ksm = ((sm-sm.mean())@q)/(n-1)/(Pyy+R)
    Kgw = ((gw-gw.mean())@q)/(n-1)/(Pyy+R)

    innov = (y + np.sqrt(R)*rng.standard_normal(n)) - Q

    for i, m in enumerate(models):
        sm_i = sm[i] + Ksm*innov[i]
        gw_i = gw[i] + Kgw*innov[i]

        m.set_value("SOIL_CONCEPTUAL_STORAGE", max(sm_i,0))
        m.gw_reservoir["storage_m"] = max(gw_i,0)

    return True

# -----------------------
# Metrics
# -----------------------

def nse(o, s):
    m = ~(np.isnan(o)|np.isnan(s))
    o,s=o[m],s[m]
    return 1 - ((o-s)**2).sum()/((o-o.mean())**2).sum()

# -----------------------
# Run
# -----------------------

def run(cat, args):
    out = Path(args.out_dir)/cat
    out.mkdir(parents=True, exist_ok=True)

    params = json.load(open(Path(args.params_dir)/cat/f"{cat}_best_params.json"))["best_parameters"]

    forcing = load_forcing(combine_forcing(cat, args.test_forcing_dir1,
                                           args.test_forcing_dir2, out))
    obs, var = load_obs(Path(args.obs_dir)/f"{cat}.csv")

    cfg = Path(args.config_file)

    spin = forcing[(forcing.date>=SPINUP_START)&(forcing.date<=SPINUP_END)]
    test = forcing[(forcing.date>=TEST_START)&(forcing.date<=TEST_END)]

    rng = np.random.default_rng(args.rng_seed)

    models, bias = build_ensemble(args.n_members, cfg, forcing, rng)

    for _, r in spin.iterrows():
        step(models, bias, r.P, r.PET, rng)
        process_noise(models, rng)

    rec, ens = [], []

    for _, r in test.iterrows():
        Q = step(models, bias, r.P, r.PET, rng)
        ens.append(Q)

        if not args.free_run:
            y = obs.get(r.date, np.nan)
            R = var.get(r.date, np.nan)
            if not np.isnan(y) and not np.isnan(R):
                enkf(models, Q, y, R, rng)

        process_noise(models, rng)

        rec.append({
            "date": r.date,
            "mean": Q.mean(),
            "std": Q.std(),
            "obs": obs.get(r.date, np.nan),
        })

    df = pd.DataFrame(rec)
    ens = np.array(ens)

    df.to_csv(out/"results.csv", index=False)

    print("NSE:", nse(df.obs.values, df["mean"].values))

    return out

# -----------------------
# CLI
# -----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cat-id")
    p.add_argument("--n-members", type=int, default=600)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--free-run", action="store_true")
    p.add_argument("--obs-dir", default=DEFAULT_OBS_DIR)
    p.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR)
    p.add_argument("--cfe-dir", default=DEFAULT_CFE_DIR)
    p.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    p.add_argument("--test-forcing-dir1", default=DEFAULT_FORCING_DIR1)
    p.add_argument("--test-forcing-dir2", default=DEFAULT_FORCING_DIR2)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    sys.path.insert(0, args.cfe_dir)
    global bmi_cfe
    import bmi_cfe as _bmi_cfe
    bmi_cfe = _bmi_cfe

    run(args.cat_id, args)

if __name__ == "__main__":
    main()
