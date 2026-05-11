# Qkrig (heldout) + troute

Routes per-catchment Qkrig (kriged streamflow, mm/h) from 21 sub-catchments of gauge 03463300
(South Toe River Near Celo, NC) directly through t-route Muskingum-Cunge. No CFE involved.
Uses held-out Qkrig files (1400-site NC pipeline).

## Contents

- `troute_environment.yml` — conda environment for reproducibility
- `run_route_troute_qkrig.py` — routing script (reads Qkrig CSVs, runs t-route MC, plots results)
- `evaluation_metrics.csv` — KGE and NSE vs USGS gauge obs
- `plots/` — time series plots (calibration period, test period, Hurricane Helene zoom)
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

## Run

```bash
conda activate troute

python3 run_route_troute_qkrig.py \
    --gpkg    /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/gage-03463300_subset.gpkg \
    --krig-dir /mnt/disk2/1400_sites_helene/catchment_ts_03463300 \
    --obs-csv  /mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv \
    --start   2020-01-01 --end 2022-12-31 \
    --out-dir  /mnt/disk2/suma_helen_poster/qkrig_troute_routing \
    --fmt kunal
```

## Results

| Period | KGE | NSE | Routed peak (m3/s) | USGS peak (m3/s) |
|---|---|---|---|---|
| Cal 2020-2022 | 0.407 | 0.527 | 208 | 396 |
| Test Oct 2023-Oct 2024 | 0.248 | 0.416 | 554 | 1886 |
