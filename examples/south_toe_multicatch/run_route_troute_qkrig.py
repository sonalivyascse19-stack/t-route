#!/usr/bin/env python3
"""
run_route_troute_qkrig.py — Route Qkrig from 21 catchments to gauge 03463300
using t-route's Muskingum-Cunge routing. No CFE involved.

Supports two CSV formats via --fmt:
  kunal  (default): columns = time, qkrig_mm_hr
                    path: /mnt/disk2/1400_sites_helene/catchment_ts_03463300
  own:              columns = datetime, qkrig
                    path: /mnt/disk2/suma_helen_poster/krig_obs_catchments_range100

Usage:
    conda activate troute

    # Kunal's files
    python3 run_route_troute_qkrig.py \
        --gpkg    /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
        --krig-dir /mnt/disk2/1400_sites_helene/catchment_ts_03463300 \
        --obs-csv  /mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv \
        --start   2020-01-01 --end 2022-12-31 \
        --out-dir  /mnt/disk2/suma_helen_poster/qkrig_troute_routing \
        --fmt kunal

    # Our own extract_krig_obs_batch.py files
    python3 run_route_troute_qkrig.py \
        --gpkg    /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
        --krig-dir /mnt/disk2/suma_helen_poster/krig_obs_catchments_range100 \
        --obs-csv  /mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv \
        --start   2020-01-01 --end 2022-12-31 \
        --out-dir  /mnt/disk2/suma_helen_poster/qkrig_troute_routing_own \
        --fmt own
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
    p.add_argument('--gpkg',     required=True, help='Hydrofabric GPKG')
    p.add_argument('--krig-dir', required=True, help='Dir with per-catchment cat-*.csv files')
    p.add_argument('--obs-csv',  required=True, help='USGS obs CSV: columns date, QObs(mm/h)')
    p.add_argument('--start',    required=True, help='Start date e.g. 2020-01-01')
    p.add_argument('--end',      required=True, help='End date e.g. 2022-12-31')
    p.add_argument('--out-dir',  required=True)
    p.add_argument('--fmt', choices=['kunal', 'own'], default='kunal',
                   help='CSV format: kunal (time, qkrig_mm_hr) or own (datetime, qkrig)')
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
        ds = int(row['to'][4:])
        connections[us] = [ds] if ds in link_set else []
    return connections


def build_param_df(fp_attr):
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


def load_usgs_obs(obs_csv, dates, total_area_m2):
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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Network ───────────────────────────────────────────────────────────────
    fp_attr, fp = read_network(args.gpkg)
    connections = build_connections(fp_attr)
    rconn = nhd_network.reverse_network(connections)

    path_func = partial(nhd_network.split_at_junction, rconn)
    reach_list = nhd_network.dfs_decomposition(rconn, path_func)
    reaches_wTypes = [(reach, 0) for reach in reach_list]
    print(f'Network: {len(fp_attr)} segments, {len(reach_list)} reaches')

    upstreams = dict(rconn)

    # ── Channel geometry ──────────────────────────────────────────────────────
    param_df = build_param_df(fp_attr)

    # ── Area map ──────────────────────────────────────────────────────────────
    area_map = {}
    for _, row in fp.iterrows():
        if row['divide_id'] and str(row['divide_id']).startswith('cat-'):
            area_map[int(row['divide_id'][4:])] = row['areasqkm'] * 1e6
    total_area_m2 = sum(area_map.values())
    print(f'Total area: {total_area_m2/1e6:.2f} km²')

    # ── Load per-catchment Qkrig CSVs ────────────────────────────────────────
    # kunal format: columns = time, qkrig_mm_hr
    # own format:   columns = datetime, qkrig
    time_col = 'time'     if args.fmt == 'kunal' else 'datetime'
    q_col    = 'qkrig_mm_hr' if args.fmt == 'kunal' else 'qkrig'
    print(f'CSV format: {args.fmt} (time_col={time_col}, q_col={q_col})')

    dfs_by_seg = {}
    for _, row in fp_attr.iterrows():
        sid = seg_int(row['link'])
        cat_id = f'cat-{sid}'
        fpath = os.path.join(args.krig_dir, f'{cat_id}.csv')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, parse_dates=[time_col])
            df = df.rename(columns={time_col: 'date', q_col: 'qkrig_mm_h'})
            df = df.set_index('date')['qkrig_mm_h']
            df = df[(df.index >= args.start) & (df.index <= args.end)]
            dfs_by_seg[sid] = df
        else:
            print(f'  Warning: {fpath} not found')

    if not dfs_by_seg:
        raise RuntimeError('No Qkrig CSV files loaded — check --krig-dir path.')

    common = sorted(set.intersection(*[set(d.index) for d in dfs_by_seg.values()]))
    dates = pd.DatetimeIndex(common)
    nts = len(dates)
    print(f'Routing {nts} timesteps: {dates[0]} → {dates[-1]}')

    # ── Build qlat (m3/s) from Qkrig ─────────────────────────────────────────
    qlat_data = {}
    Q_krig_total = np.zeros(nts)
    for sid, df in dfs_by_seg.items():
        area = area_map.get(sid, 0.0)
        vals = df.reindex(dates).values
        qlat_data[sid] = np.nan_to_num(vals) / 1000.0 / 3600.0 * area
        Q_krig_total += qlat_data[sid]

    qlat_df = pd.DataFrame(qlat_data, index=range(nts)).T.sort_index()
    qlat_df = qlat_df.reindex(param_df.index, fill_value=0.0)

    peak_qlat = qlat_df.values.sum(axis=0).max()
    peak_t = int(qlat_df.values.sum(axis=0).argmax())
    print(f'Peak total lateral inflow: {peak_qlat:.3f} m3/s at t={peak_t} ({dates[peak_t]})')

    # ── Initial conditions ────────────────────────────────────────────────────
    n_segs = len(param_df)
    q0_df = pd.DataFrame(
        np.zeros((n_segs, 3), dtype='float32'),
        index=param_df.index,
        columns=['qu0', 'qd0', 'h0']
    )

    # ── USGS obs ──────────────────────────────────────────────────────────────
    obs_q = load_usgs_obs(args.obs_csv, dates, total_area_m2)

    # ── Empty DA / reservoir arrays ───────────────────────────────────────────
    e1i   = np.zeros(0, dtype='int32')
    e1f   = np.zeros(0, dtype='float32')
    e2f   = np.zeros((0, nts), dtype='float32')
    e00f32 = np.zeros((0, 0), dtype='float32')
    e00f64 = np.zeros((0, 0), dtype='float64')
    e00i32 = np.zeros((0, 0), dtype='int32')

    model_start_time = dates[0].strftime('%Y-%m-%d_%H:%M:%S')

    # ── Run t-route routing ───────────────────────────────────────────────────
    print('Running compute_network_structured ...')
    results = compute_network_structured(
        nts, DT, QTS_SUBDIVISIONS,
        reaches_wTypes,
        upstreams,
        param_df.index.values.astype('int64'),
        param_df.columns.values,
        param_df.values,
        q0_df.values.astype('float32'),
        qlat_df.values.astype('float32'),
        [],
        e00f64,
        {},
        e00i32,
        False,
        model_start_time,
        e2f, e1i, e1i, e1i, e1f, e1f, 0.0,
        e2f, e1i, e1f, e1f, e1f, e1f, e1f,
        e2f, e1i, e1f, e1f, e1f, e1f, e1f,
        e2f, e1i, e1i, [], e1i, e1i, e1f, e1i, e1i,
        e1i, e1i, e1f, e1i, e1f, e1i, e1i, e00f32,
    )

    # ── Parse results ─────────────────────────────────────────────────────────
    seg_ids = np.asarray(results[0])
    fvd = np.asarray(results[1])
    terminal_pos = int(np.where(seg_ids == TERMINAL_INT)[0][0])
    print(f'fvd shape: {fvd.shape}, terminal seg {TERMINAL_INT} at pos {terminal_pos}')

    Q_time_major = fvd[terminal_pos, 0::3]
    Q_var_major  = fvd[terminal_pos, :nts]
    print(f'Time-major Q peak: {Q_time_major.max():.3f} m3/s')
    print(f'Var-major  Q peak: {Q_var_major.max():.3f} m3/s')

    Q_outlet = Q_var_major if Q_var_major.max() > Q_time_major.max() else Q_time_major

    # ── Metrics ───────────────────────────────────────────────────────────────
    kge = compute_kge(obs_q, Q_outlet)
    nse = compute_nse(obs_q, Q_outlet)
    kge_krig = compute_kge(Q_krig_total, Q_outlet)
    nse_krig = compute_nse(Q_krig_total, Q_outlet)
    print(f'\nRouted Qkrig:        peak={Q_outlet.max():.3f} m3/s at {dates[Q_outlet.argmax()]}')
    print(f'USGS obs:            peak={np.nanmax(obs_q):.3f} m3/s')
    print(f'Area-weighted Qkrig: peak={Q_krig_total.max():.3f} m3/s')
    print(f'KGE (routed Qkrig vs USGS obs):          {kge:.4f}')
    print(f'NSE (routed Qkrig vs USGS obs):          {nse:.4f}')
    print(f'KGE (routed Qkrig vs area-wtd Qkrig):   {kge_krig:.4f}')
    print(f'NSE (routed Qkrig vs area-wtd Qkrig):   {nse_krig:.4f}')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    period_tag = f"{args.start[:7].replace('-','')}_{args.end[:7].replace('-','')}"
    out_df = pd.DataFrame({
        'date':               dates,
        'Q_routed_m3s':       Q_outlet,
        'Q_usgs_m3s':         obs_q,
        'Q_krig_total_m3s':   Q_krig_total,
    })
    csv_path = os.path.join(args.out_dir, f'routed_qkrig_troute_{period_tag}.csv')
    out_df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    kge_krig_usgs = compute_kge(obs_q, Q_krig_total)
    nse_krig_usgs = compute_nse(obs_q, Q_krig_total)
    print(f'KGE (area-wtd Qkrig vs USGS obs): {kge_krig_usgs:.4f}')
    print(f'NSE (area-wtd Qkrig vs USGS obs): {nse_krig_usgs:.4f}')

    # ── Plot: USGS obs + Area-weighted Qkrig + Routed Qkrig ──────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, obs_q, 'k-', linewidth=0.8, label='USGS obs (gauge 03463300)', alpha=0.85)
    ax.plot(dates, Q_krig_total, color='darkorange', linewidth=0.8, linestyle='--',
            label='Area-weighted Qkrig', alpha=0.7)
    ax.plot(dates, Q_outlet, color='steelblue', linewidth=0.8,
            label='Routed Qkrig — t-route Muskingum-Cunge', alpha=0.85)
    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title(
        f'Routed discharge at gauge 03463300 (South Toe River) — {args.start} to {args.end}\n'
        f'KGE={kge:.3f}  NSE={nse:.3f}  |  t-route Muskingum-Cunge',
        fontsize=11
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, f'routed_qkrig_troute_{period_tag}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {plot_path}')


if __name__ == '__main__':
    main()
