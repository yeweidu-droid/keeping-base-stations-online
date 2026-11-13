# BRIDGE: code and data

Source code, input data, and per-run result files for the paper

> Keeping Base Stations Online After Disasters: Joint Scheduling Under Battery Countdown

BRIDGE schedules mobile generators, UAV base stations, base-station energy saving, and shared storage in microgrids after a power-grid outage. This package contains everything needed to check the numbers in the paper and to rerun the experiments.

## Contents

```
README.md                 this file
DATA_SOURCES.md           origin and license of the input data
requirements.txt          Python dependencies
verify_quick.py           35-second end-to-end check
sim/                      simulator, algorithms, experiment drivers
  scenario.py             scenario construction from the OpenStreetMap extracts
  bridge.py               BRIDGE and all baselines and ablation variants
  online.py               rolling-horizon operation under estimation errors
  runlib.py               instance cache and algorithm dispatch
  experiments.py          experiment definitions and multi-process driver
  milp_bench.py           exact MILP on the small instances
  certificate.py          a-posteriori upper bound (R) and the spillover check
  refine_rank.py          rank correlation of the refinement screen
  sleep_split.py          UAV coverage while the covered station sleeps
  plots.py                figures from the stored results
  fetch_osm.py            download script for the OpenStreetMap extracts
  run_all.sh              runs every experiment
  data/                   raw OpenStreetMap extracts (JSON)
  results/                one JSON file per run, for every experiment
  README_zh.md            the authors' original notes (Chinese)
```

Comments inside the source files are in Chinese. The file and function names follow the terms of the paper.

## Requirements

Python 3.10 or later with NumPy, SciPy, and Matplotlib. All linear and mixed-integer programs are solved by HiGHS through SciPy, so no separate solver is needed. The results were produced with Python 3.12.12, NumPy 2.5.1, SciPy 1.18.1, and Matplotlib 3.11.0.

```
pip install -r requirements.txt
```

## Quick check

```
python3 verify_quick.py            # Luding, seed 0
python3 verify_quick.py xuwen 3    # any scenario and seed from 0 to 19
```

The script rebuilds one instance from the raw OpenStreetMap extract, reruns GEN-EDF, GEN-LS, IND, SEQ, and BRIDGE, and compares the weighted service ratio with the stored result. Every algorithm is deterministic for a given seed, so the values agree to floating-point precision. The first run of an instance takes about 20 seconds to build the travel-time tables, which are then cached in `sim/cache/`.

## Figures

```
python3 sim/plots.py         # Chinese labels, written to ../paper_zh/figs/
python3 sim/plots.py en      # English ablation and microgrid-size figures, written to ../paper/figs/
```

`plots.py` reads `sim/results/` and writes vector PDFs. It does not write LaTeX tables.

## Rerunning the experiments

```
cd sim
NW=16 ./run_all.sh                       # everything, NW worker processes
python3 experiments.py main 16           # or one experiment at a time
python3 milp_bench.py 600 16             # exact MILP, 600 s per instance
python3 certificate.py 16                # upper bound on the full-size instances
python3 certificate.py 16 small          # upper bound on the small instances
```

A task whose result file already exists is skipped. To rerun an experiment, delete its folder under `sim/results/`. On a 16-core machine the experiments in `experiments.py` take about one hour in total, and the MILP benchmark takes up to 600 seconds per instance.

## Which experiment is behind which result

| Paper item | Experiment | Result files | Seeds |
|---|---|---|---|
| Table 4, Fig. 4, Fig. 5, Fig. 8, Tables S4 and S6, Fig. S2 | `main` | `results/main/` | 20 |
| Stronger baselines (Section 6.4) | `base` | `results/base/` | 20 |
| Fig. 6 | `mgsize` | `results/mgsize/` | 10 |
| Table 5 | `mgnear` | `results/mgnear/` | 10 |
| Table 6, Table S1 | `milp_bench.py` | `results/milp/` | 10 per size |
| Table 7 | `certificate.py` | `results/certificate.json` | 20 |
| Fig. 7 | `online`, `road` | `results/online/`, `results/road/` | 10 |
| Table S2 | `demand`, `combo` | `results/demand/`, `results/combo/` | 10 |
| Table S3 | `golden` | `results/golden/` | 20 |
| Fig. 9, Fig. S1 | `sens`, `block` | `results/sens/`, `results/block/` | 10 |
| Planning time (supplementary material) | `runtime` | `results/runtime/` | 3 per size |
| Table S5 | `uavrole`, `sleep_split.py` | `results/uavrole/`, `results/sleep_split.json` | 20 |
| Rank correlation of the refinement screen | `refine_rank.py` | `results/refine_rank.json` | 20 |

## Algorithm names in the code

| Paper | Code |
|---|---|
| GEN-EDF, GEN-LS, IND, SEQ, BRIDGE | the same names |
| No station energy layer | `w/o EA` |
| No energy saving | `w/o ES` |
| No shared storage | `w/o MG` |
| No UAVs | `w/o UAV` |
| No route optimization | `w/o routing` |
| Marginal gain without group-level price | `w/o grpprice` |
| No joint neighborhood refinement | `w/o refine` |
| No UAV shadow price | `w/o price` |
| Energy saving ignores UAVs | `w/o coupling` |
| No alternation | `w/o BCA` |
| SEQ+refine, SEQ+refine* | `SEQ+R`, `SEQ+R*` |
| Simulated annealing, 1 and 5 times the running time | `SA-1x`, `SA-5x` |

In the result files, `SR` is the weighted service ratio, `WO_kuh` and `WO` are the weighted disconnected user-hours in thousands, `J` is the objective, `crit_out_h` is the equivalent disconnection time of key stations in hours, `dark_site_h` is the number of station-hours with zero service, and `jain` is the Jain index.

## Input data

The road networks, settlements, and facilities of the two study areas come from OpenStreetMap. Station sites are obtained by clustering the settlements. Demand, power values, battery endurance, and damaged road segments are generated from the random seed by `sim/scenario.py`. See `DATA_SOURCES.md` for the origin and the license of the OpenStreetMap extracts.
