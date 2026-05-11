#!/usr/bin/env python3
"""
run_route_save_per_seg.py
--------------------------------------------------------------------------------
Same routing as run_route_troute.py but additionally saves per-segment Q
for all 21 reaches across the full period AND a Helene-only window.

Outputs (in --out-dir):
  per_seg_Q_{mode}_{period}_helene.csv  -- hourly Q (m3/s) for all 21 segs, Sep 25-30 2024
  per_seg_Q_{mode}_{period}_peak.csv   -- single-row peak Q per segment across full period

Usage:
  conda activate troute

  # CFE + troute (Run 3)
  python run_route_save_per_seg.py \
      --mode cfe \
      --gpkg /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
      --results-dir /mnt/disk2/suma_helen_poster/catchment_results_range100_run3 \
      --period test \
      --out-dir /mnt/disk2/suma_helen_poster/alternative_plots

  # Qkrig + troute
  python run_route_save_per_seg.py \
      --mode qkrig \
      --gpkg /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
      --krig-dir /mnt/disk2/1400_sites_helene/catchment_ts_03463300 \
      --period test \
      --out-dir /mnt/disk2/suma_helen_poster/alternative_plots
"""

import argparse
import sqlite3
import os
import re
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd

import troute.nhd_network as nhd_network
from troute.routing.fast_reach.mc_reach import compute_network_structured

TERMINAL_INT   = 1016283
DT             = 3600.0
QTS_SUBDIVISIONS = 1
HELENE_START   = pd.Timestamp('2024-09-25')
HELENE_END     = pd.Timestamp('2024-09-30 23:00:00')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['cfe', 'qkrig'], required=True)
    p.add_argument('--gpkg', required=True)
    p.add_argument('--results-dir', default=None, help='CFE results dir (mode=cfe)')
    p.add_argument('--krig-dir',    default=None, help='Qkrig CSV dir  (mode=qkrig)')
    p.add_argument('--period', choices=['cal', 'test'], default='test')
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def seg_int(wb_id):
    return int(wb_id[3:])


def read_network(gpkg):
    con = sqlite3.connect(gpkg)
    fp_attr = pd.read_sql(
        'SELECT link, "to", BtmWdth, TopWdth, TopWdthCC, n, nCC, ChSlp, So, Length_m '
        'FROM "flowpath-attributes"', con)
    fp = pd.read_sql('SELECT divide_id, areasqkm FROM flowpaths', con)
    con.close()
    return fp_attr, fp


def build_connections(fp_attr):
    link_set = {seg_int(r) for r in fp_attr['link']}
    connections = {}
    for _, row in fp_attr.iterrows():
        us = seg_int(row['link'])
        ds = int(row['to'][4:])
        connections[us] = [ds] if ds in link_set else []
    return connections


def build_param_df(fp_attr):
    rows = []
    for _, row in fp_attr.iterrows():
        rows.append({
            'seg_id': seg_int(row['link']),
            'dt': float(DT), 'bw': float(row['BtmWdth']),
            'tw': float(row['TopWdth']), 'twcc': float(row['TopWdthCC']),
            'dx': float(row['Length_m']), 'n': float(row['n']),
            'ncc': float(row['nCC']), 'cs': float(row['ChSlp']),
            's0': float(row['So']), 'alt': 0.0,
        })
    return pd.DataFrame(rows).set_index('seg_id').sort_index().astype('float32')


def load_cfe_qlat(fp_attr, fp, results_dir, period):
    suffix = '_cal_results.csv' if period == 'cal' else '_test_results.csv'
    area_map = {}
    for _, row in fp.iterrows():
        if row['divide_id'] and str(row['divide_id']).startswith('cat-'):
            area_map[int(row['divide_id'][4:])] = row['areasqkm'] * 1e6
    total_area = sum(area_map.values())

    dfs = {}
    for _, row in fp_attr.iterrows():
        sid = seg_int(row['link'])
        cat_id = f'cat-{sid}'
        fpath = os.path.join(results_dir, cat_id, f'{cat_id}{suffix}')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, parse_dates=['date']).set_index('date')
            dfs[sid] = df
        else:
            print(f'  [WARN] {fpath} not found')

    common = sorted(set.intersection(*[set(d.index) for d in dfs.values()]))
    dates = pd.DatetimeIndex(common)
    nts = len(dates)

    qlat_data = {}
    for sid, df in dfs.items():
        area = area_map.get(sid, 0.0)
        qlat_data[sid] = df.loc[dates, 'sim_mm_h'].values / 1000.0 / 3600.0 * area

    return dates, nts, qlat_data, total_area


def load_qkrig_qlat(fp_attr, fp, krig_dir, period):
    area_map = {}
    for _, row in fp.iterrows():
        if row['divide_id'] and str(row['divide_id']).startswith('cat-'):
            area_map[int(row['divide_id'][4:])] = row['areasqkm'] * 1e6
    total_area = sum(area_map.values())

    krig_path = Path(krig_dir)
    dfs = {}
    for fpath in sorted(krig_path.glob("cat-*.csv")):
        m = re.search(r"cat-(\d+)\.csv$", fpath.name)
        if not m:
            continue
        sid = int(m.group(1))
        try:
            df = pd.read_csv(fpath, parse_dates=["time"], index_col="time",
                             usecols=["time", "qkrig_mm_hr"])
            dfs[sid] = df
        except Exception as e:
            print(f'  [WARN] {fpath.name}: {e}')

    common = sorted(set.intersection(*[set(d.index) for d in dfs.values()]))
    if period == 'cal':
        common = [d for d in common if pd.Timestamp('2020-01-01') <= d <= pd.Timestamp('2022-12-31 23:00')]
    else:
        common = [d for d in common if pd.Timestamp('2023-10-01') <= d <= pd.Timestamp('2024-10-01')]

    dates = pd.DatetimeIndex(common)
    nts = len(dates)

    qlat_data = {}
    for sid, df in dfs.items():
        area = area_map.get(sid, 0.0)
        vals = df.loc[dates, 'qkrig_mm_hr'].values
        vals = np.where(np.isnan(vals) | (vals < 0), 0.0, vals)
        qlat_data[sid] = vals / 1000.0 / 3600.0 * area

    return dates, nts, qlat_data, total_area


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fp_attr, fp = read_network(args.gpkg)
    connections = build_connections(fp_attr)
    rconn = nhd_network.reverse_network(connections)
    path_func = partial(nhd_network.split_at_junction, rconn)
    reach_list = nhd_network.dfs_decomposition(rconn, path_func)
    reaches_wTypes = [(reach, 0) for reach in reach_list]
    upstreams = dict(rconn)
    param_df = build_param_df(fp_attr)

    if args.mode == 'cfe':
        dates, nts, qlat_data, total_area = load_cfe_qlat(fp_attr, fp, args.results_dir, args.period)
    else:
        dates, nts, qlat_data, total_area = load_qkrig_qlat(fp_attr, fp, args.krig_dir, args.period)

    print(f'Routing {nts} timesteps: {dates[0]} -> {dates[-1]}  (mode={args.mode})')

    qlat_df = pd.DataFrame(qlat_data, index=range(nts)).T.sort_index()
    qlat_df = qlat_df.reindex(param_df.index, fill_value=0.0)

    n_segs = len(param_df)
    q0_df = pd.DataFrame(np.zeros((n_segs, 3), dtype='float32'),
                         index=param_df.index, columns=['qu0', 'qd0', 'h0'])

    e1i = np.zeros(0, dtype='int32')
    e1f = np.zeros(0, dtype='float32')
    e2f = np.zeros((0, nts), dtype='float32')
    e00f32 = np.zeros((0, 0), dtype='float32')
    e00f64 = np.zeros((0, 0), dtype='float64')
    e00i32 = np.zeros((0, 0), dtype='int32')
    model_start_time = dates[0].strftime('%Y-%m-%d_%H:%M:%S')

    print('Running compute_network_structured ...')
    results = compute_network_structured(
        nts, DT, QTS_SUBDIVISIONS, reaches_wTypes, upstreams,
        param_df.index.values.astype('int64'), param_df.columns.values,
        param_df.values, q0_df.values.astype('float32'),
        qlat_df.values.astype('float32'),
        [], e00f64, {}, e00i32, False, model_start_time,
        e2f, e1i, e1i, e1i, e1f, e1f, 0.0,
        e2f, e1i, e1f, e1f, e1f, e1f, e1f,
        e2f, e1i, e1f, e1f, e1f, e1f, e1f,
        e2f, e1i, e1i, [], e1i, e1i, e1f, e1i, e1i,
        e1i, e1i, e1f, e1i, e1f, e1i, e1i, e00f32,
    )

    seg_ids = np.asarray(results[0])
    fvd     = np.asarray(results[1])
    terminal_pos = int(np.where(seg_ids == TERMINAL_INT)[0][0])

    # Determine layout (time-major vs variable-major)
    Q_time_major = fvd[terminal_pos, 0::3]
    Q_var_major  = fvd[terminal_pos, :nts]
    tm_max = float(np.nanmax(Q_time_major)) if not np.all(np.isnan(Q_time_major)) else 0.0
    vm_max = float(np.nanmax(Q_var_major))  if not np.all(np.isnan(Q_var_major))  else 0.0
    if vm_max > tm_max:
        Q_all = fvd[:, :nts]   # variable-major: (n_segs, nts)
        print(f'Using variable-major layout (terminal peak={vm_max:.3f} m3/s)')
    else:
        Q_all = fvd[:, 0::3]   # time-major: (n_segs, nts)
        print(f'Using time-major layout (terminal peak={tm_max:.3f} m3/s)')

    terminal_q = Q_all[terminal_pos]
    print(f'Terminal seg peak Q: {np.nanmax(terminal_q):.3f} m3/s at {dates[np.nanargmax(terminal_q)]}')

    # -- Save peak Q per segment (full period) ---------------------------------
    peak_df = pd.DataFrame({
        'seg_id': seg_ids,
        'peak_Q_m3s': np.nanmax(Q_all, axis=1),
    })
    peak_path = os.path.join(args.out_dir, f'per_seg_Q_{args.mode}_{args.period}_peak.csv')
    peak_df.to_csv(peak_path, index=False)
    print(f'Saved peak per-seg Q -> {peak_path}')

    # -- Save hourly Q for Helene window ---------------------------------------
    helene_mask = (dates >= HELENE_START) & (dates <= HELENE_END)
    helene_dates = dates[helene_mask]
    helene_Q = Q_all[:, helene_mask]   # (n_segs, n_helene_hours)

    helene_df = pd.DataFrame(
        helene_Q.T,
        index=helene_dates,
        columns=[f'wb-{sid}' for sid in seg_ids]
    )
    helene_path = os.path.join(args.out_dir, f'per_seg_Q_{args.mode}_{args.period}_helene.csv')
    helene_df.to_csv(helene_path)
    print(f'Saved Helene per-seg Q ({len(helene_dates)} hours) -> {helene_path}')


if __name__ == '__main__':
    main()
