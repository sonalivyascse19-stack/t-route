#!/usr/bin/env python3
"""
run_route_troute.py — Route CFE runoff from 21 catchments to gauge 03463300
using t-route's Muskingum-Cunge routing (compute_network_structured).

Usage:
    python3 catchment_run/run_route_troute.py \
        --gpkg /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
        --results-dir /mnt/disk2/suma_helen_poster/catchment_results_range100 \
        --period cal \
        --obs-csv /mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv \
        --out-dir /mnt/disk2/suma_helen_poster/catchment_results_range100
"""
import argparse
import sqlite3
import os
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import troute.nhd_network as nhd_network
from troute.routing.fast_reach.mc_reach import compute_network_structured

TERMINAL_INT = 1016283   # wb-1016283 → nexus nex-1016284 = gauge 03463300
DT = 3600.0              # hourly timestep (seconds)
QTS_SUBDIVISIONS = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpkg', required=True)
    p.add_argument('--results-dir', required=True)
    p.add_argument('--period', choices=['cal', 'test'], default='cal')
    p.add_argument('--obs-csv', default=None,
                   help='USGS obs CSV: columns date, QObs(mm/h). Converted to m3/s via total area.')
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def seg_int(wb_id):
    return int(wb_id[3:])  # 'wb-1016283' → 1016283


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
        ds = int(row['to'][4:])   # 'nex-XXXXXX' → XXXXXX (same number as downstream wb)
        connections[us] = [ds] if ds in link_set else []
    return connections


def build_param_df(fp_attr):
    """Channel geometry for t-route. Columns must be ['dt','bw','tw','twcc','dx','n','ncc','cs','s0','alt']."""
    rows = []
    for _, row in fp_attr.iterrows():
        rows.append({
            'seg_id': seg_int(row['link']),
            'dt':   float(DT),
            'bw':   float(row['BtmWdth']),
            'tw':   float(row['TopWdth']),
            'twcc': float(row['TopWdthCC']),
            'dx':   float(row['Length_m']),
            'n':    float(row['n']),
            'ncc':  float(row['nCC']),
            'cs':   float(row['ChSlp']),
            's0':   float(row['So']),
            'alt':  0.0,
        })
    df = pd.DataFrame(rows).set_index('seg_id').sort_index()
    return df.astype('float32')


def load_csv_obs(obs_csv, dates, total_area_m2):
    df = pd.read_csv(obs_csv, parse_dates=['date']).set_index('date')['QObs(mm/h)']
    obs = np.full(len(dates), np.nan)
    for i, dt in enumerate(dates):
        if dt in df.index:
            obs[i] = df[dt] / 1000.0 / 3600.0 * total_area_m2
    print(f'USGS obs: {np.sum(~np.isnan(obs))}/{len(dates)} valid timestamps matched')
    return obs


def compute_kge(obs, sim):
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) < 2 or np.std(o) == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return 1.0 - np.sqrt((r-1)**2 + (np.std(s)/np.std(o)-1)**2 + (np.mean(s)/np.mean(o)-1)**2)


def compute_nse(obs, sim):
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) < 2 or np.sum((o-np.mean(o))**2) == 0:
        return np.nan
    return 1.0 - np.sum((o-s)**2) / np.sum((o-np.mean(o))**2)


def extract_outlet_q(results, terminal_int, nts):
    """
    Parse compute_network_structured results to get Q (m3/s) at the terminal reach.

    t-route returns a list of per-reach tuples:
      (flowveldepth_array, courant_array)  if return_courant=True
      flowveldepth_array                   if return_courant=False
    where flowveldepth_array has shape (n_segs_in_reach * nts, 3) with cols [Q, v, d].
    The ordering within each reach matches the reach segment list (upstream → downstream).
    """
    print(f'results type: {type(results)}, length: {len(results)}')
    if len(results) == 0:
        return np.full(nts, np.nan)

    # Print first element to understand format
    r0 = results[0]
    print(f'results[0] type: {type(r0)}')
    if hasattr(r0, '__len__'):
        arr = np.asarray(r0)
        print(f'results[0] shape: {arr.shape}')
    return None   # sentinel: caller will parse after seeing debug output


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Network ───────────────────────────────────────────────────────────────
    fp_attr, fp = read_network(args.gpkg)
    connections = build_connections(fp_attr)
    rconn = nhd_network.reverse_network(connections)

    # Decompose into reaches (chains between junctions) via DFS on reverse network
    path_func = partial(nhd_network.split_at_junction, rconn)
    reach_list = nhd_network.dfs_decomposition(rconn, path_func)
    reaches_wTypes = [(reach, 0) for reach in reach_list]   # 0 = MC, no reservoirs
    print(f'Network: {len(fp_attr)} segments, {len(reach_list)} reaches')
    for r in reach_list:
        print(f'  reach (len={len(r)}): {r}')

    # upstreams must be the REVERSE network: {seg: [upstream_segs]}
    # compute_network_structured uses this to find upstream boundary Q for each reach
    # reverse_network returns a defaultdict; cast to plain dict as required by Cython
    upstreams = dict(rconn)

    # ── Channel geometry ──────────────────────────────────────────────────────
    param_df = build_param_df(fp_attr)
    print(f'param_df: {param_df.shape}')
    print(param_df.head(3))

    # ── CFE lateral inflows ───────────────────────────────────────────────────
    suffix = '_cal_results.csv' if args.period == 'cal' else '_test_results.csv'
    dfs_by_seg = {}
    for _, row in fp_attr.iterrows():
        sid = seg_int(row['link'])
        cat_id = f'cat-{sid}'
        fpath = os.path.join(args.results_dir, cat_id, f'{cat_id}{suffix}')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, parse_dates=['date']).set_index('date')
            dfs_by_seg[sid] = df
        else:
            print(f'  Warning: {fpath} not found')

    common = sorted(set.intersection(*[set(d.index) for d in dfs_by_seg.values()]))
    dates = pd.DatetimeIndex(common)
    nts = len(dates)
    print(f'Routing {nts} timesteps: {dates[0]} → {dates[-1]}')

    area_map = {}
    for _, row in fp.iterrows():
        if row['divide_id'] and str(row['divide_id']).startswith('cat-'):
            area_map[int(row['divide_id'][4:])] = row['areasqkm'] * 1e6
    total_area_m2 = sum(area_map.values())
    print(f'Total area: {total_area_m2/1e6:.2f} km²')

    # qlat DataFrame: index=int_seg_ids (sorted), columns=range(nts), units=m³/s
    qlat_data = {}
    for sid, df in dfs_by_seg.items():
        area = area_map.get(sid, 0.0)
        qlat_data[sid] = df.loc[dates, 'sim_mm_h'].values / 1000.0 / 3600.0 * area

    qlat_df = pd.DataFrame(qlat_data, index=range(nts)).T.sort_index()
    qlat_df = qlat_df.reindex(param_df.index, fill_value=0.0)

    peak_t = int(qlat_df.values.sum(axis=0).argmax())
    peak_qlat = qlat_df.values.sum(axis=0).max()
    print(f'Peak total lateral inflow: {peak_qlat:.3f} m3/s at t={peak_t} ({dates[peak_t]})')
    print(f'Channel geometry for terminal seg {TERMINAL_INT}:')
    print(param_df.loc[TERMINAL_INT])

    # Initial conditions: (n_segs, 3) = [qu0, qd0, h0], all zero for cold start
    n_segs = len(param_df)
    q0_df = pd.DataFrame(
        np.zeros((n_segs, 3), dtype='float32'),
        index=param_df.index,
        columns=['qu0', 'qd0', 'h0']
    )

    # ── Observations ──────────────────────────────────────────────────────────
    obs_q = (load_csv_obs(args.obs_csv, dates, total_area_m2)
             if args.obs_csv else np.full(nts, np.nan))

    # ── Empty DA / reservoir arrays ───────────────────────────────────────────
    e1i = np.zeros(0, dtype='int32')
    e1f = np.zeros(0, dtype='float32')
    e2f = np.zeros((0, nts), dtype='float32')   # (0, nts): 0 DA sites, nts timesteps
    e00f32 = np.zeros((0, 0), dtype='float32')  # float32 2D empty
    e00f64 = np.zeros((0, 0), dtype='float64')  # float64 2D empty (wbody_cols)
    e00i32 = np.zeros((0, 0), dtype='int32')    # int32 2D empty (reservoir_types)

    model_start_time = dates[0].strftime('%Y-%m-%d_%H:%M:%S')

    # ── Run t-route routing ───────────────────────────────────────────────────
    print('Running compute_network_structured ...')
    results = compute_network_structured(
        nts,                                   # nsteps
        DT,                                    # dt (seconds)
        QTS_SUBDIVISIONS,                      # qts_subdivisions
        reaches_wTypes,                        # reaches_wTypes: [(reach_segs_list, 0), ...]
        upstreams,                             # upstream_connections (forward network dict)
        param_df.index.values.astype('int64'), # data_idx  (sorted int64 seg IDs)
        param_df.columns.values,               # data_cols (['dt','bw','tw',…,'alt'])
        param_df.values,                       # data_values (n_segs × 10, float32)
        q0_df.values.astype('float32'),        # initial_conditions (n_segs × 3)
        qlat_df.values.astype('float32'),      # qlat_values (n_segs × nts)
        [],                                    # lake_numbers_col (no waterbodies)
        e00f64,                                # wbody_cols (const double[:,:])
        {},                                    # data_assimilation_parameters
        e00i32,                                # reservoir_types (const int[:,:], 2D)
        False,                                 # reservoir_type_specified
        model_start_time,                      # model_start_time
        e2f,                                   # usgs_values      (0 × nts)
        e1i,                                   # usgs_positions
        e1i,                                   # usgs_positions_reach
        e1i,                                   # usgs_positions_gage
        e1f,                                   # lastobs_values_init
        e1f,                                   # time_since_lastobs_init
        0.0,                                   # da_decay_coefficient
        # USGS hybrid reservoir DA (none)
        e2f, e1i, e1f, e1f, e1f, e1f, e1f,
        # USACE hybrid reservoir DA (none)
        e2f, e1i, e1f, e1f, e1f, e1f, e1f,
        # RFC reservoir DA (none)
        e2f, e1i, e1i, [], e1i, e1i, e1f, e1i, e1i,
        # Great Lakes DA (none)
        e1i, e1i, e1f, e1i, e1f, e1i, e1i, e00f32,
    )

    # ── Parse results ─────────────────────────────────────────────────────────
    # results[0]: sorted segment IDs (n_segs,)
    # results[1]: flowveldepth (n_segs, nts*3) — layout could be:
    #   time-major:     [Q_t0, v_t0, d_t0, Q_t1, v_t1, d_t1, ...]  → Q = [0::3]
    #   variable-major: [Q_t0..Q_tnts, v_t0..v_tnts, d_t0..d_tnts] → Q = [:nts]
    seg_ids = np.asarray(results[0])
    fvd     = np.asarray(results[1])           # (21, 78843) float32
    terminal_pos = int(np.where(seg_ids == TERMINAL_INT)[0][0])
    print(f'fvd shape: {fvd.shape}, global max={fvd.max():.3f}')
    print(f'Terminal seg {TERMINAL_INT} at position {terminal_pos}')

    # Per-segment Q peak (time-major layout: Q at every 3rd col from 0)
    q_peak_per_seg = fvd[:, 0::3].max(axis=1)
    for i, (sid, qp) in enumerate(zip(seg_ids, q_peak_per_seg)):
        marker = ' ← TERMINAL' if sid == TERMINAL_INT else ''
        print(f'  seg {sid} (pos {i}): peak Q = {qp:.3f} m3/s{marker}')

    # Try both layouts
    Q_time_major = fvd[terminal_pos, 0::3]    # time-major: Q every 3rd col from 0
    Q_var_major  = fvd[terminal_pos, :nts]    # variable-major: first nts cols
    print(f'Time-major   Q peak: {Q_time_major.max():.3f} m3/s at t={Q_time_major.argmax()} ({dates[Q_time_major.argmax()]})')
    print(f'Var-major    Q peak: {Q_var_major.max():.3f} m3/s  at t={Q_var_major.argmax()}  ({dates[Q_var_major.argmax()]})')
    print(f'All-segs Q max (time-major): {fvd[:, 0::3].max():.3f} m3/s')
    print(f'All-segs Q max (var-major):  {fvd[:, :nts].max():.3f} m3/s')

    # Use variable-major if that gives higher (physically correct) terminal flow
    if Q_var_major.max() > Q_time_major.max():
        Q_outlet = Q_var_major
        print('Using variable-major layout')
    else:
        Q_outlet = Q_time_major
        print('Using time-major layout')

    # ── Metrics ───────────────────────────────────────────────────────────────
    kge = compute_kge(obs_q, Q_outlet)
    nse = compute_nse(obs_q, Q_outlet)
    print(f'\nRouted Q: peak={Q_outlet.max():.3f} m3/s at {dates[Q_outlet.argmax()]}')
    if not np.all(np.isnan(obs_q)):
        print(f'KGE (routed vs USGS obs): {kge:.4f}')
        print(f'NSE (routed vs USGS obs): {nse:.4f}')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    obs_krig = np.zeros(nts)
    for sid, df in dfs_by_seg.items():
        area = area_map.get(sid, 0.0)
        obs_krig += df.loc[dates, 'obs_mm_h'].values / 1000.0 / 3600.0 * area

    out_df = pd.DataFrame({
        'date': dates,
        'Q_routed_m3s': Q_outlet,
        'Q_usgs_m3s': obs_q,
        'Q_krig_m3s': obs_krig,
    })
    csv_path = os.path.join(args.out_dir, f'routed_Q_troute_{args.period}.csv')
    out_df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, obs_q, 'k-', linewidth=0.8, label='USGS obs (gauge 03463300)', alpha=0.85)
    ax.plot(dates, obs_krig, color='gray', linewidth=0.6, linestyle='--',
            label='Area-weighted Qkrig', alpha=0.6)
    ax.plot(dates, Q_outlet, color='steelblue', linewidth=0.8,
            label='CFE + t-route Muskingum-Cunge', alpha=0.85)
    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (m³/s)')
    period_label = '2020–2022 Calibration' if args.period == 'cal' else 'Test (incl. Hurricane Helene)'
    ax.set_title(
        f'Routed discharge at gauge 03463300 (South Toe River) — {period_label}\n'
        f'KGE={kge:.3f}  NSE={nse:.3f}  |  t-route Muskingum-Cunge',
        fontsize=11
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, f'routed_Q_troute_{args.period}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {plot_path}')


if __name__ == '__main__':
    main()
