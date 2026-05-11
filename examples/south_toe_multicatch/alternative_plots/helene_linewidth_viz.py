#!/usr/bin/env python3
"""
helene_linewidth_viz.py
--------------------------------------------------------------------------------
Visualise Hurricane Helene (Sep 25-30 2024) routing results using
linewidth-encoded discharge on the river network.

River reaches are drawn with line thickness AND colour proportional to routed
Q (m3/s). Both figures use the same blue colormap so they are directly
comparable. Catchment polygons shown as light background context.

Requires per_seg_Q_{mode}_test_helene.csv from run_route_save_per_seg.py.

Usage:
  python helene_linewidth_viz.py \
      --gpkg /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
      --cfe-csv  /mnt/disk2/suma_helen_poster/alternative_plots/per_seg_Q_cfe_test_helene.csv \
      --qkrig-csv /mnt/disk2/suma_helen_poster/alternative_plots/per_seg_Q_qkrig_test_helene.csv \
      --out-dir  /mnt/disk2/suma_helen_poster/alternative_plots
"""

import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

GAGE_LAT =  35.83138889
GAGE_LON = -82.18416670
TERMINAL_INT = 1016283

DAYS = [pd.Timestamp(f'2024-09-{d:02d}') for d in range(25, 31)]
PHASE = {
    25: ('Pre-landfall',    '#e08214'),
    26: ('Pre-landfall',    '#e08214'),
    27: ('Landfall / peak', '#d6191b'),
    28: ('Post-landfall',   '#2166ac'),
    29: ('Post-landfall',   '#2166ac'),
    30: ('Post-landfall',   '#2166ac'),
}

STREAM_LAYER_NAMES = ['flowpaths', 'streams', 'network', 'flowlines', 'hydroedge']

# Single blue colormap for both figures — truncated to skip pale range
CMAP = LinearSegmentedColormap.from_list(
    'blues_dark', cm.Blues(np.linspace(0.30, 1.00, 256)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpkg',      required=True)
    p.add_argument('--cfe-csv',   required=True)
    p.add_argument('--qkrig-csv', required=True)
    p.add_argument('--out-dir',   required=True)
    return p.parse_args()


def load_flowpaths(gpkg):
    available = fiona.listlayers(gpkg)
    for name in STREAM_LAYER_NAMES:
        if name in available:
            gdf = gpd.read_file(gpkg, layer=name).to_crs(4326)
            if not gdf.empty:
                print(f'  Loaded flowpaths from layer: {name}  ({len(gdf)} features)')
                return gdf
    raise RuntimeError('No flowpath layer found in GPKG')


def get_seg_id(row):
    for col in ['id', 'link', 'divide_id', 'wb_id']:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col])
            if val.startswith('wb-'):
                return int(val[3:])
    return None


def load_per_seg_q(csv_path):
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def make_panel(ax, divides, flowpaths, q_by_seg, day, norm, phase_label, phase_color):
    start = day
    end   = day + pd.Timedelta(hours=23)
    day_q = q_by_seg.loc[start:end] if start in q_by_seg.index else pd.DataFrame()

    divides.plot(ax=ax, facecolor='#f0f4f8', edgecolor='#cccccc',
                 linewidth=0.3, zorder=1)

    if day_q.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f'Sep {day.day}\nno data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='#888888')
        return

    peak_q = {}
    for col in day_q.columns:
        try:
            sid = int(col.replace('wb-', ''))
            peak_q[sid] = float(day_q[col].max())
        except Exception:
            pass

    for _, row in flowpaths.iterrows():
        sid = get_seg_id(row)
        if sid is None:
            continue
        q_val = peak_q.get(sid, 0.0)
        if np.isnan(q_val):
            q_val = 0.0

        # Linewidth: 1.0 (min) to 7.5 (max), power=0.25 boosts small tributaries
        lw    = 1.0 + 6.5 * (norm(q_val) ** 0.25)
        color = CMAP(norm(q_val))

        gpd.GeoDataFrame(geometry=[row.geometry], crs=4326).plot(
            ax=ax, color=color, linewidth=lw, zorder=3)

    divides.dissolve().plot(ax=ax, facecolor='none', edgecolor='#d6191b',
                            linewidth=1.2, linestyle='--', zorder=4)

    ax.plot(GAGE_LON, GAGE_LAT, marker='o', markersize=8,
            color='#f7e900', markeredgecolor='#1a1a1a', markeredgewidth=0.8,
            zorder=8, clip_on=True)
    ax.text(GAGE_LON + 0.008, GAGE_LAT + 0.005, '03463300',
            fontsize=5.5, color='#f7e900', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='#1a1a1a', ec='none', alpha=0.65),
            zorder=9, clip_on=True)

    terminal_q = peak_q.get(TERMINAL_INT, np.nan)
    ax.set_title(f'Sep {day.day}  |  outlet peak = {terminal_q:.0f} m³/s',
                 fontsize=8.5, fontweight='bold', pad=4)

    ax.text(0.02, 0.03, phase_label, transform=ax.transAxes, fontsize=7,
            va='bottom', ha='left', color=phase_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec=phase_color, lw=0.6, alpha=0.88), zorder=6)
    ax.set_axis_off()


def make_figure(divides, flowpaths, q_by_seg, title, metrics_text, out_path, q_max):
    norm  = mcolors.Normalize(vmin=0, vmax=max(q_max, 1.0))

    fig = plt.figure(figsize=(13, 9), facecolor='white')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.12, wspace=0.06,
                            left=0.03, right=0.88,
                            top=0.88, bottom=0.04)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    for ax, day in zip(axes, DAYS):
        phase_label, phase_color = PHASE[day.day]
        make_panel(ax, divides, flowpaths, q_by_seg, day,
                   norm, phase_label, phase_color)

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.65])
    sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Routed Q  (m³/s)', fontsize=9, labelpad=7)
    ticks = np.linspace(0, q_max, 6)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([f'{v:.0f}' for v in ticks], fontsize=7.5)
    cb.ax.tick_params(width=0.5, length=3)
    cb.outline.set_linewidth(0.5)

    # Linewidth legend
    legend_color = mcolors.to_hex(CMAP(0.85))
    legend_elements = [
        Line2D([0], [0], color=legend_color, linewidth=0.5 + 7.5*np.sqrt(0.1), label='Low Q'),
        Line2D([0], [0], color=legend_color, linewidth=0.5 + 7.5*np.sqrt(0.5), label='Mid Q'),
        Line2D([0], [0], color=legend_color, linewidth=0.5 + 7.5*np.sqrt(1.0), label='High Q'),
    ]
    fig.legend(handles=legend_elements, loc='lower right',
               bbox_to_anchor=(0.995, 0.02), fontsize=7,
               title='Line width = Q', title_fontsize=7, framealpha=0.9)

    # Metrics box
    fig.text(0.905, 0.93, metrics_text, fontsize=7.5, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', fc='#f8f8f8',
                       ec='#444444', lw=1.0, alpha=0.95))

    # USGS gauge legend
    cbar_ax.plot(0.5, -0.05, marker='o', markersize=6, color='#f7e900',
                 markeredgecolor='#1a1a1a', markeredgewidth=0.7,
                 transform=cbar_ax.transAxes, clip_on=False, zorder=10)
    cbar_ax.text(0.5, -0.075, 'USGS\ngage', ha='center', va='top',
                 fontsize=6, transform=cbar_ax.transAxes)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.97)
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    print(f'Saved -> {out_path}')
    plt.close(fig)


def main():
    args = parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'\nReading GeoPackage: {args.gpkg}')
    divides   = gpd.read_file(args.gpkg, layer='divides').to_crs(4326)
    flowpaths = load_flowpaths(args.gpkg)

    print(f'\nLoading CFE per-seg Q: {args.cfe_csv}')
    cfe_q = load_per_seg_q(args.cfe_csv)

    print(f'Loading Qkrig per-seg Q: {args.qkrig_csv}')
    qkrig_q = load_per_seg_q(args.qkrig_csv)

    # Shared scale across both experiments so figures are directly comparable
    shared_q_max = max(float(cfe_q.max().max()), float(qkrig_q.max().max()))
    print(f'Shared Q max: {shared_q_max:.1f} m³/s')

    # --- Figure 1: Qkrig + troute ---
    print('\nGenerating Qkrig + troute linewidth figure ...')
    make_figure(
        divides, flowpaths, qkrig_q,
        title=('South Toe River  [held-out]  --  Qkrig + t-route  |  '
               'Hurricane Helene  Sep 25-30 2024\n'
               'Line width = routed Q per reach (m³/s)   *   '
               'Marker = USGS gage 03463300 (near Celo, NC)'),
        metrics_text='Qkrig (held-out) + t-route\nCal KGE=0.407  NSE=0.527\nTest KGE=0.248  NSE=0.416\nHelene routed peak=554 m³/s\nUSGS peak=1886 m³/s',
        out_path=f'{args.out_dir}/helene_linewidth_qkrig_troute.png',
        q_max=shared_q_max,
    )

    # --- Figure 2: CFE + troute ---
    print('\nGenerating CFE + troute linewidth figure ...')
    make_figure(
        divides, flowpaths, cfe_q,
        title=('South Toe River  [held-out]  --  CFE + t-route  |  '
               'Hurricane Helene  Sep 25-30 2024\n'
               'Line width = routed Q per reach (m³/s)   *   '
               'Marker = USGS gage 03463300 (near Celo, NC)'),
        metrics_text='Qkrig (held-out) + CFE + t-route\nCal KGE=0.359  NSE=0.372\nTest KGE=0.167  NSE=0.417\nHelene routed peak=688 m³/s\nUSGS peak=1886 m³/s',
        out_path=f'{args.out_dir}/helene_linewidth_cfe_troute.png',
        q_max=shared_q_max,
    )


if __name__ == '__main__':
    main()
