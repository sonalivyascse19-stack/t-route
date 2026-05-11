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

Create the conda environment from the included `troute_environment.yml`, then build t-route:

```bash
conda env create -f examples/south_toe_multicatch/troute_environment.yml
conda activate troute
cd /path/to/this/repo

cd src/troute-network && python setup.py build_ext --inplace && cd ../..
cd src/troute-routing && python setup.py build_ext --inplace && cd ../..
export PYTHONPATH=$(pwd)/src/troute-routing:$PYTHONPATH
```

## Changes to Sonam's ngiab branch

Only two files were modified from the base `ngiab` branch:

**1. `src/troute-network/setup.py`** — comment out `rfc_reservoirs` extension (requires
`./libs/bind_rfc.a` static library which is not in the repo) and make Fortran detection optional:

```diff
-rfc_reservoirs = Extension(
-    "troute.network.reservoirs.rfc.rfc",
-    ...
-)
+#rfc_reservoirs = Extension(
+#    "troute.network.reservoirs.rfc.rfc",
+#    ...
+#)

-ext_modules = [reach, levelpool_reservoirs, rfc_reservoirs, musk]
+ext_modules = [reach, levelpool_reservoirs, musk]

-result = subprocess.run([fc, '--version'], stdout=subprocess.PIPE)
-if "GNU" in result: fcompiler_type = 'gnu95'
-elif "Intel" in result: fcompiler_type = 'intel'
-else: raise Exception("Could not identify fortran compiler!")
+fcompiler_type = None
+if fc:
+    result = subprocess.run([fc, '--version'], stdout=subprocess.PIPE)
+    if "GNU" in result: fcompiler_type = 'gnu95'
+    elif "Intel" in result: fcompiler_type = 'intel'
```

**2. `src/troute-routing/setup.py`** — same Fortran detection fix (same diff as above, no RFC change needed here).

### Run routing

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

**Held-in gauge (Run 2 — range100 NPZ Qkrig):**

| File | Description |
|---|---|
| `plots/routed_Q_troute_cal.png` | Routed vs USGS — calibration period 2020–2022 (KGE=0.52, NSE=0.46) |
| `plots/routed_Q_troute_test.png` | Routed vs USGS — test period Oct 2023–Oct 2024 (KGE=0.26, NSE=0.49) |
| `plots/helene_routing_comparison.png` | Hurricane Helene zoom: USGS vs Qkrig vs CFE+t-route, routed peak=699 m³/s |

**Held-out gauge (Run 3 — Kunal's NC Qkrig):**

| File | Description |
|---|---|
| `plots/run3_routed_Q_troute_cal.png` | Routed vs USGS — calibration period 2020–2022 (KGE=0.359, NSE=0.372) |
| `plots/run3_routed_Q_troute_test.png` | Routed vs USGS — test period Oct 2023–Oct 2024 (KGE=0.167, NSE=0.417) |
| `plots/run3_helene_zoom_routing.png` | Hurricane Helene zoom: USGS vs Qkrig vs CFE+t-route, routed peak=688 m³/s |
