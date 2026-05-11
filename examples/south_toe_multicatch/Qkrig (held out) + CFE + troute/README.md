# Qkrig (held out) + CFE + troute

CFE is calibrated against held-out Qkrig (Kunal's NC pipeline, gauge 03463300 excluded).
CFE runoff from 21 sub-catchments is then routed through t-route Muskingum-Cunge to gauge 03463300
(South Toe River Near Celo, NC).

## Contents

- `troute_environment.yml` — conda environment for reproducibility
- `calibrate_catchment_nwm.py` — DDS calibration script (N=1000 iterations, against held-out Qkrig)
- `run_route_troute.py` — routing script (reads CFE NPZ outputs, runs t-route MC, plots results)
- `params/` — best calibration parameters for all 21 catchments (cat-XXXXXXX_best_params.json)
- `calibration_and_eval_metrics.csv` — per-catchment calibration KGE + routing evaluation metrics
- `plots/` — time series plots for held-out run (calibration period, test period, Hurricane Helene zoom)
- `hydrofabric/` — hydrofabric visualisation (coming soon)

## Setup

```bash
conda env create -f troute_environment.yml
conda activate troute

cd /path/to/t-route
cd src/troute-network && python setup.py build_ext --inplace && cd ../..
cd src/troute-routing && python setup.py build_ext --inplace && cd ../..
export PYTHONPATH=$(pwd)/src/troute-routing:$PYTHONPATH
```

## GPU paths

| Resource | Path |
|---|---|
| Hydrofabric GPKG | `/mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg` |
| CFE results (held-out run) | `<path to Run 3 CFE results on GPU>` |
| USGS obs CSV | `/mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv` |

## Run routing

```bash
conda activate troute

python3 run_route_troute.py \
    --gpkg        /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
    --results-dir <path to Run 3 CFE results on GPU> \
    --period      cal \
    --obs-csv     /mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv \
    --out-dir     /mnt/disk2/suma_helen_poster/catchment_results_range100
```

## Results

Calibration KGE range across 21 catchments: 0.72-0.83 (see `calibration_and_eval_metrics.csv`).

| Period | KGE | NSE | Routed peak (m3/s) | USGS peak (m3/s) |
|---|---|---|---|---|
| Cal 2020-2022 | 0.359 | 0.372 | - | 396 |
| Test Oct 2023-Oct 2024 | 0.167 | 0.417 | 688 | 1886 |
