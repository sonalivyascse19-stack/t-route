# South Toe River — 21-Catchment MC Routing

Routes CFE runoff from 21 sub-catchments of gauge **03463300** (South Toe River Near Celo, NC)
through t-route Muskingum-Cunge to the terminal reach `wb-1016283` (nexus `nex-1016284`).

CFE was calibrated against area-weighted Qkrig observations (our own range100 NPZ files).
See [calibrate-cfe](https://github.com/NWC-CUAHSI-Summer-Institute/calibrate-cfe) for calibration scripts and parameters.

## Results

| Period | KGE vs USGS | NSE vs USGS | Routed peak (m³/s) | USGS peak (m³/s) |
|---|---|---|---|---|
| Cal (2020–2022) | 0.52 | 0.46 | 196 | 396 |
| Test (Oct 2023–Oct 2024, incl. Hurricane Helene) | 0.26 | 0.49 | 699 | 1886 |

Helene peak timing: routed at Sep 27 17:00, USGS at Sep 27 14:00 (3 hours late).

Underestimation of Helene peak (~63%) is driven by CFE being calibrated against Qkrig,
not direct USGS gauge obs — not a routing issue.

## GPU paths

| Resource | Path |
|---|---|
| Hydrofabric GPKG | `/mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg` |
| CFE results (Run 2, cal) | `/mnt/disk2/suma_helen_poster/catchment_results_range100/` |
| USGS obs CSV | `/mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv` |
| Output CSVs | `routed_Q_troute_{cal,test}.csv` |

## Setup

### 1. Install t-route (RFC extension disabled)

The `rfc_reservoirs` build extension in `src/troute-network/setup.py` has been commented out
because it requires Fortran `netcdff`/`netcdf` libraries. Build without it:

```bash
conda activate troute
cd /home/svyas/t-route-ngiab

pip install -e src/troute-network/
pip install -e src/troute-routing/
```

### 2. Run routing

```bash
conda activate troute

# Calibration period (2020–2022)
python3 examples/south_toe_multicatch/run_route_troute.py \
    --gpkg        /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
    --results-dir /mnt/disk2/suma_helen_poster/catchment_results_range100 \
    --period      cal \
    --obs-csv     /mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv \
    --out-dir     /mnt/disk2/suma_helen_poster/catchment_results_range100

# Test period (Oct 2023–Oct 2024)
python3 examples/south_toe_multicatch/run_route_troute.py \
    --gpkg        /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
    --results-dir /mnt/disk2/suma_helen_poster/catchment_results_range100 \
    --period      test \
    --obs-csv     /mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv \
    --out-dir     /mnt/disk2/suma_helen_poster/catchment_results_range100
```

Outputs: `routed_Q_troute_{cal,test}.csv` and `routed_Q_troute_{cal,test}.png`

## Plots

| File | Description |
|---|---|
| `plots/routed_Q_troute_cal.png` | Routed vs USGS — calibration period 2020–2022 |
| `plots/routed_Q_troute_test.png` | Routed vs USGS — test period Oct 2023–Oct 2024 |
| `plots/helene_routing_comparison.png` | Hurricane Helene zoom: USGS vs Qkrig vs CFE+t-route |
