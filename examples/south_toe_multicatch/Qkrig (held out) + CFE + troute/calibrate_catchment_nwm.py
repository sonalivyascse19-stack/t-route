"""
Calibrate CFE for a single catchment using NWM retro forcing + kriging obs.
Optionally run test period using NWM operational forcing.

Forcing (training):  per-catchment NWM retro CSV from extract_nwm_forcing.py
                     columns: time, UGRD_10maboveground, precip_rate, DSWRF_surface,
                              TMP_2maboveground, SPFH_2maboveground, VGRD_10maboveground,
                              DLWRF_surface, PRES_surface, APCP_surface
Forcing (testing):   per-catchment NWM operational CSVs (two dirs: 2023_2024_feb, 2024_feb_2025_sep)
Obs:                 per-catchment kriging CSV from extract_krig_obs_batch.py
                     columns: datetime, qkrig, variance  (qkrig in mm/h)

Time splits:
  Spinup (cal):   2019-01-01 00:00 → 2019-12-31 23:00
  Calibration:    2020-01-01 00:00 → 2022-12-31 23:00
  Spinup (test):  2023-02-01 00:00 → 2023-09-30 23:00
  Test (Helene):  2023-10-01 00:00 → 2024-10-31 23:00

Usage:
  python3 calibrate_catchment_nwm.py \
    --cat-id cat-1016300 \
    --forcing-dir  /mnt/disk2/suma_helen_poster/nwm_retro_catchment_forcings \
    --obs-dir      /mnt/disk2/suma_helen_poster/krig_obs_catchments \
    --cfe-dir      /mnt/disk2/suma_helen_poster/cfe_py \
    --config-file  /mnt/disk2/suma_helen_poster/run_gpu/cat_03463300_bmi_config_cfe.json \
    --param-bounds /mnt/disk2/suma_helen_poster/run_gpu/CFE_parameter_bounds.json \
    --out-dir      /mnt/disk2/suma_helen_poster/catchment_results \
    --test-forcing-dir1 /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/output_03463300_nwmoperational/03463300/2023_2024_feb/forcings \
    --test-forcing-dir2 /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/output_03463300_nwmoperational/03463300/2024_feb_2025_sep/forcings \
    --N 1000

To run all 21 catchments in parallel:
  for cat in cat-1016279 cat-1016280 cat-1016281 cat-1016282 cat-1016283 \
             cat-1016300 cat-1016301 cat-1016302 cat-1016303 cat-1016304 \
             cat-1016305 cat-1016306 cat-1016307 cat-1016308 cat-1016309 \
             cat-1016310 cat-1016311 cat-1016312 cat-1016313 cat-1016314 \
             cat-1016315; do
    nohup python3 calibrate_catchment_nwm.py --cat-id $cat ... > logs/${cat}.log 2>&1 &
  done
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import spotpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ALBEDO   = 0.20
ALPHA_PT = 1.26

TIME_SPLIT = {
    "spinup-for-calibration": {
        "start": "2019-01-01 00:00:00",
        "end":   "2019-12-31 23:00:00",
    },
    "calibration": {
        "start": "2020-01-01 00:00:00",
        "end":   "2022-12-31 23:00:00",
    },
    "spinup-for-testing": {
        "start": "2023-02-01 00:00:00",
        "end":   "2023-09-30 23:00:00",
    },
    "testing": {
        "start": "2023-10-01 00:00:00",
        "end":   "2024-10-31 23:00:00",
    },
}

# Set at runtime
CAT_ID            = None
FORCING_FILE      = None
TEST_FORCING_FILE = None
OBS_FILE          = None
CFE_CONFIG_FILE   = None
PARAM_BOUNDS_FILE = None
OUT_DIR           = None
bmi_cfe           = None


def priestley_taylor_pet(srad_wm2, T_kelvin, alpha=ALPHA_PT):
    """PT-PET from shortwave radiation + temperature. Returns mm/h."""
    T = T_kelvin - 273.15
    delta = 4098 * (0.6108 * np.exp(17.27 * T / (T + 237.3))) / (T + 237.3) ** 2
    gamma = 0.0638
    Rn_mj = np.maximum((1.0 - ALBEDO) * srad_wm2, 0.0) * 0.0036
    lam = 2.501 - 0.002361 * T
    pet = alpha * (delta / (delta + gamma)) * Rn_mj / lam
    return np.maximum(pet, 0.0)


def load_forcing(path=None):
    """Load NWM forcing CSV (retro or operational) and add PET column. Returns DataFrame."""
    path = path or FORCING_FILE
    df = pd.read_csv(path)
    df = df.rename(columns={"time": "date", "APCP_surface": "total_precipitation"})
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['potential_evaporation'] = priestley_taylor_pet(
        df['DSWRF_surface'].values,
        df['TMP_2maboveground'].values
    )
    return df


def load_test_forcing():
    """Load NWM operational forcing CSV. APCP_surface is in kg/m²/s → convert to mm/h."""
    df = pd.read_csv(TEST_FORCING_FILE)
    df['date'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['total_precipitation'] = df['APCP_surface'] * 3600.0  # kg/m²/s → mm/h
    df['potential_evaporation'] = priestley_taylor_pet(
        df['DSWRF_surface'].values,
        df['TMP_2maboveground'].values
    )
    return df


def load_obs_for_period(forcing_df, period_start, period_end):
    """
    Merge kriging obs onto forcing dates for a given period.
    Returns (dates array, obs array with NaN for gaps).
    """
    mask = (forcing_df['date'] >= period_start) & (forcing_df['date'] <= period_end)
    period_dates = forcing_df[mask][['date']].reset_index(drop=True)

    obs_raw = pd.read_csv(OBS_FILE)
    obs_raw['date'] = pd.to_datetime(obs_raw['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    obs_raw = obs_raw.rename(columns={"qkrig": "obs_mm_h"})

    merged = period_dates.merge(obs_raw[['date', 'obs_mm_h']], on='date', how='left')
    return merged['date'].values, merged['obs_mm_h'].values


class SpotpySetup(object):

    def __init__(self, parameter_bounds):
        self.parameter_bounds = parameter_bounds

        with open(CFE_CONFIG_FILE) as f:
            cfg = json.load(f)

        optguess = {
            'bb':             cfg['soil_params']['bb'],
            'smcmax':         cfg['soil_params']['smcmax'],
            'satdk':          cfg['soil_params']['satdk'],
            'slop':           cfg['soil_params']['slop'],
            'max_gw_storage': cfg['max_gw_storage'],
            'expon':          cfg['expon'],
            'Cgw':            cfg['Cgw'],
            'K_lf':           cfg['K_lf'],
            'K_nash':         cfg['K_nash'],
            'scheme':         1,
        }

        self.params = [
            spotpy.parameter.Uniform(
                name,
                details['lower_bound'],
                details['upper_bound'],
                optguess=optguess[name]
            )
            for name, details in parameter_bounds.items()
        ]

        self.df_forcing = load_forcing()
        self.eval_dates, self.obs_data = load_obs_for_period(
            self.df_forcing,
            TIME_SPLIT['calibration']['start'],
            TIME_SPLIT['calibration']['end']
        )

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):

        def custom_load_forcing(self_cfe):
            df = load_forcing()
            self_cfe.forcing_data = df.rename(columns={"date": "time"})

        with open(CFE_CONFIG_FILE) as f:
            cfg = json.load(f)

        cfg['forcing_file']          = FORCING_FILE
        cfg['soil_params']['bb']     = vector['bb']
        cfg['soil_params']['smcmax'] = vector['smcmax']
        cfg['soil_params']['satdk']  = vector['satdk']
        cfg['slop']                  = vector['slop']
        cfg['max_gw_storage']        = vector['max_gw_storage']
        cfg['expon']                 = vector['expon']
        cfg['Cgw']                   = vector['Cgw']
        cfg['K_lf']                  = vector['K_lf']
        cfg['K_nash']                = vector['K_nash']
        cfg['partition_scheme']      = "Schaake" if vector['scheme'] <= 0.5 else "Xinanjiang"

        tmp_cfg = str(OUT_DIR / f'{CAT_ID}_bmi_config_temp.json')
        with open(tmp_cfg, 'w') as f:
            json.dump(cfg, f)

        model = bmi_cfe.BMI_CFE(cfg_file=tmp_cfg)
        model.load_forcing_file = custom_load_forcing.__get__(model)
        model.initialize()

        df = self.df_forcing

        # Spinup
        sp_mask = (df['date'] >= TIME_SPLIT['spinup-for-calibration']['start']) & \
                  (df['date'] <= TIME_SPLIT['spinup-for-calibration']['end'])
        df_sp = df[sp_mask]
        for p, e in zip(df_sp['total_precipitation'], df_sp['potential_evaporation']):
            model.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', p / 1000)
            model.set_value('water_potential_evaporation_flux', e / 1000 / 3600)
            model.update()

        # Calibration
        cal_mask = (df['date'] >= TIME_SPLIT['calibration']['start']) & \
                   (df['date'] <= TIME_SPLIT['calibration']['end'])
        df_cal = df[cal_mask]
        outputs = model.get_output_var_names()
        out_lists = {o: [] for o in outputs}

        for p, e in zip(df_cal['total_precipitation'], df_cal['potential_evaporation']):
            model.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', p / 1000)
            model.set_value('water_potential_evaporation_flux', e / 1000 / 3600)
            model.update()
            for o in outputs:
                out_lists[o].append(model.get_value(o))

        model.finalize()
        return np.array(out_lists['land_surface_water__runoff_depth']) * 1000  # m/h → mm/h

    def evaluation(self, evaldates=False):
        if evaldates:
            return [pd.Timestamp(d) for d in self.eval_dates]
        return self.obs_data

    def objectivefunction(self, simulation, evaluation, params=None):
        mask = ~np.isnan(evaluation)
        if mask.sum() == 0:
            return np.nan
        return spotpy.objectivefunctions.kge(evaluation[mask], simulation[mask])


def run_testing_period(best_param_dict):
    """Run CFE over test period using NWM operational forcing."""

    def custom_load_forcing(self_cfe):
        df = load_test_forcing()
        self_cfe.forcing_data = df.rename(columns={"date": "time"})

    with open(CFE_CONFIG_FILE) as f:
        cfg = json.load(f)

    cfg['forcing_file']          = TEST_FORCING_FILE
    cfg['soil_params']['bb']     = best_param_dict['bb']
    cfg['soil_params']['smcmax'] = best_param_dict['smcmax']
    cfg['soil_params']['satdk']  = best_param_dict['satdk']
    cfg['slop']                  = best_param_dict['slop']
    cfg['max_gw_storage']        = best_param_dict['max_gw_storage']
    cfg['expon']                 = best_param_dict['expon']
    cfg['Cgw']                   = best_param_dict['Cgw']
    cfg['K_lf']                  = best_param_dict['K_lf']
    cfg['K_nash']                = best_param_dict['K_nash']
    cfg['partition_scheme']      = "Schaake" if best_param_dict['scheme'] <= 0.5 else "Xinanjiang"

    tmp_cfg = str(OUT_DIR / f'{CAT_ID}_bmi_config_temp_test.json')
    with open(tmp_cfg, 'w') as f:
        json.dump(cfg, f)

    model = bmi_cfe.BMI_CFE(cfg_file=tmp_cfg)
    model.load_forcing_file = custom_load_forcing.__get__(model)
    model.initialize()

    df = load_test_forcing()

    # Spinup for test
    sp_mask = (df['date'] >= TIME_SPLIT['spinup-for-testing']['start']) & \
              (df['date'] <= TIME_SPLIT['spinup-for-testing']['end'])
    for p, e in zip(df[sp_mask]['total_precipitation'], df[sp_mask]['potential_evaporation']):
        model.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', p / 1000)
        model.set_value('water_potential_evaporation_flux', e / 1000 / 3600)
        model.update()

    # Test period
    t_mask = (df['date'] >= TIME_SPLIT['testing']['start']) & \
             (df['date'] <= TIME_SPLIT['testing']['end'])
    df_test = df[t_mask]
    outputs = model.get_output_var_names()
    out_lists = {o: [] for o in outputs}

    for p, e in zip(df_test['total_precipitation'], df_test['potential_evaporation']):
        model.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', p / 1000)
        model.set_value('water_potential_evaporation_flux', e / 1000 / 3600)
        model.update()
        for o in outputs:
            out_lists[o].append(model.get_value(o))

    model.finalize()

    sim_test = np.array(out_lists['land_surface_water__runoff_depth']) * 1000  # m/h → mm/h
    test_dates = pd.to_datetime(df_test['date'].values)

    # Load kriging obs for test period
    obs_raw = pd.read_csv(OBS_FILE)
    obs_raw['date'] = pd.to_datetime(obs_raw['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    obs_raw = obs_raw.rename(columns={"qkrig": "obs_mm_h"})
    test_dates_df = pd.DataFrame({'date': test_dates.strftime('%Y-%m-%d %H:%M:%S')})
    merged = test_dates_df.merge(obs_raw[['date', 'obs_mm_h']], on='date', how='left')
    obs_test = merged['obs_mm_h'].values

    mask = ~np.isnan(obs_test)
    kge_test = spotpy.objectivefunctions.kge(obs_test[mask], sim_test[mask])
    nse_test = spotpy.objectivefunctions.nashsutcliffe(obs_test[mask], sim_test[mask])
    print(f"Test KGE: {kge_test:.4f} | Test NSE: {nse_test:.4f}")

    df_out = pd.DataFrame({
        'date':        test_dates.strftime('%Y-%m-%d %H:%M:%S'),
        'sim_mm_h':    sim_test,
        'obs_mm_h':    obs_test,
        'precip_mm_h': df_test['total_precipitation'].values,
    })
    df_out.to_csv(OUT_DIR / f'{CAT_ID}_test_results.csv', index=False)

    # Plot test period
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    ax1.plot(test_dates, sim_test, 'tomato', lw=1.5, label='simulated')
    ax1.plot(test_dates, obs_test, 'k', lw=1, label='observed (kriging)')
    ax1.set_ylabel('Discharge (mm/h)')
    ax1.set_title(f'{CAT_ID} | Test (Oct 2023–Oct 2024) | KGE={kge_test:.4f} | NSE={nse_test:.4f}')
    ax1.legend()
    ax1_twin = ax1.twinx()
    ax1_twin.plot(test_dates, df_test['total_precipitation'].values, 'steelblue', lw=0.8, alpha=0.5)
    ax1_twin.set_ylim([80, 0])
    ax1_twin.set_ylabel('Precip (mm/h)')

    helene = (test_dates >= pd.Timestamp('2024-09-20')) & (test_dates <= pd.Timestamp('2024-10-05'))
    ax2.plot(test_dates[helene], sim_test[helene], 'tomato', lw=2, label='simulated')
    ax2.plot(test_dates[helene], obs_test[helene], 'k', lw=1.5, label='observed')
    ax2.set_ylabel('Discharge (mm/h)')
    ax2.set_title('Helene Zoom: Sep 20 – Oct 5, 2024')
    ax2.legend()
    ax2_twin = ax2.twinx()
    ax2_twin.bar(test_dates[helene], df_test['total_precipitation'].values[helene],
                 color='steelblue', alpha=0.4, width=0.04)
    ax2_twin.set_ylim([50, 0])
    ax2_twin.set_ylabel('Precip (mm/h)')

    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{CAT_ID}_test_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    return kge_test, nse_test


def main():
    global CAT_ID, FORCING_FILE, TEST_FORCING_FILE, OBS_FILE
    global CFE_CONFIG_FILE, PARAM_BOUNDS_FILE, OUT_DIR, bmi_cfe

    parser = argparse.ArgumentParser()
    parser.add_argument('--cat-id',            required=True,  help='Catchment ID, e.g. cat-1016300')
    parser.add_argument('--forcing-dir',        required=True,  help='Dir with per-catchment NWM retro CSVs')
    parser.add_argument('--obs-dir',            required=True,  help='Dir with per-catchment kriging obs CSVs')
    parser.add_argument('--cfe-dir',            required=True,  help='Path to cfe_py directory')
    parser.add_argument('--config-file',        required=True,  help='Base CFE BMI config JSON')
    parser.add_argument('--param-bounds',       required=True,  help='Parameter bounds JSON')
    parser.add_argument('--out-dir',            required=True,  help='Output directory')
    parser.add_argument('--test-forcing-dir1',  default=None,   help='NWM operational forcings dir 1 (2023-Feb2024)')
    parser.add_argument('--test-forcing-dir2',  default=None,   help='NWM operational forcings dir 2 (Feb2024-2025)')
    parser.add_argument('--N',                  type=int, default=1000, help='DDS iterations')
    parser.add_argument('--test-only',          action='store_true', help='Skip calibration, run test only')
    args = parser.parse_args()

    CAT_ID            = args.cat_id
    FORCING_FILE      = os.path.join(args.forcing_dir, f'{CAT_ID}.csv')
    OBS_FILE          = os.path.join(args.obs_dir,     f'{CAT_ID}.csv')
    CFE_CONFIG_FILE   = args.config_file
    PARAM_BOUNDS_FILE = args.param_bounds
    OUT_DIR           = Path(args.out_dir) / CAT_ID
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build combined test forcing file if both dirs provided
    if args.test_forcing_dir1 and args.test_forcing_dir2:
        f1 = os.path.join(args.test_forcing_dir1, f'{CAT_ID}.csv')
        f2 = os.path.join(args.test_forcing_dir2, f'{CAT_ID}.csv')
        if os.path.exists(f1) and os.path.exists(f2):
            df1 = pd.read_csv(f1)
            df2 = pd.read_csv(f2)
            combined = pd.concat([df1, df2], ignore_index=True)
            combined = combined.drop_duplicates(subset='time').sort_values('time')
            TEST_FORCING_FILE = str(OUT_DIR / f'{CAT_ID}_nwm_operational_combined.csv')
            combined.to_csv(TEST_FORCING_FILE, index=False)
        else:
            print(f"Warning: test forcing files not found for {CAT_ID}, skipping test period")

    sys.path.insert(0, args.cfe_dir)
    import bmi_cfe as _bmi_cfe
    bmi_cfe = _bmi_cfe

    with open(PARAM_BOUNDS_FILE) as f:
        parameter_bounds = json.load(f)

    best_params_file = OUT_DIR / f'{CAT_ID}_best_params.json'

    if not args.test_only:
        print(f"Calibrating {CAT_ID} | N={args.N}")
        print(f"Forcing:  {FORCING_FILE}")
        print(f"Obs:      {OBS_FILE}")
        print(f"Cal period: {TIME_SPLIT['calibration']['start']} → {TIME_SPLIT['calibration']['end']}")

        instance = SpotpySetup(parameter_bounds)

        np.random.seed(0)
        sampler = spotpy.algorithms.dds(instance, dbname=f'dds_{CAT_ID}', dbformat='ram')
        sampler.sample(args.N)
        results = sampler.getdata()

        best_params     = spotpy.analyser.get_best_parameterset(results)
        best_param_dict = {name: val for name, val in zip(parameter_bounds.keys(), best_params[0])}
        obj_values      = results['like1']
        best_kge        = float(np.nanmax(obj_values))
        best_idx        = np.where(obj_values == np.nanmax(obj_values))[0][0]
        best_sim        = np.array([v for v in spotpy.analyser.get_modelruns(results[best_idx])])

        print(f"\n{CAT_ID} — Best KGE: {best_kge:.4f}")
        print(f"Best params: {best_param_dict}")

        out = {
            "catchment_id":    CAT_ID,
            "best_kge":        best_kge,
            "best_parameters": best_param_dict,
        }
        with open(best_params_file, 'w') as f:
            json.dump(out, f, indent=4)

        # Save calibration timeseries
        dates = instance.evaluation(evaldates=True)
        df_out = pd.DataFrame({
            'date':     [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
            'sim_mm_h': best_sim,
            'obs_mm_h': instance.obs_data,
        })
        df_out.to_csv(OUT_DIR / f'{CAT_ID}_cal_results.csv', index=False)

        # Plot calibration (first year)
        _, ax = plt.subplots(figsize=(16, 5))
        ax.plot(dates[:8760], best_sim[:8760], 'tomato', lw=1.5, label='simulated')
        ax.plot(dates[:8760], instance.obs_data[:8760], 'k', lw=1, label='observed (kriging)')
        ax.set_ylabel('Discharge (mm/h)')
        ax.set_title(f'{CAT_ID} | Calibration first year (2020) | KGE={best_kge:.4f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{CAT_ID}_cal_plot.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Results saved to {OUT_DIR}")

    # Run test period if test forcing is available
    if TEST_FORCING_FILE and os.path.exists(TEST_FORCING_FILE):
        if best_params_file.exists():
            with open(best_params_file) as f:
                saved = json.load(f)
            best_param_dict = saved['best_parameters']
            print(f"\nRunning test period (Oct 2023 – Oct 2024)...")
            run_testing_period(best_param_dict)
        else:
            print("No best params file found. Run calibration first.")
    else:
        print("No test forcing available, skipping test period.")


if __name__ == '__main__':
    main()
