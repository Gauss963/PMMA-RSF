# TS0304-TS0320: RSF velocity calibration and loading-rate sweep

TS0278 is the sole template. Geometry, material, 16 MPa normal loading, 2.45 mm
final shear displacement, middle/loading VW law, leading VS law, mesh and 10 ns
time step retain their TS0278 values unless noted below.

## Velocity-law calibration

The middle/loading-zone direct effect stays at `a = 0.005`. The dynamic
calibration velocity changes from 2000 to 200 mm/s, so

```
b = a + (0.8 - 0.45) / ln(200 / 1e-4) = 0.029123527228205267.
```

The RSF characteristic slip changes from 0.0003614447313307937 to
0.00042852936846183833 mm. This preserves the *calibrated ageing-law velocity
step* breakdown work `sigma_n * b * D_c * I(V_dyn/V_init)` used to anchor
TS0278 to its CZM fracture energy. It does not assert that the realized dynamic
fracture energy will be identical; that must be measured from each run.

## Loading-rate design

TS0304-TS0319 vary only half-cosine ramp time (and the minimum simulation
duration needed to observe it). Their peak imposed speeds span 25.66-200 mm/s
with piecewise logarithmic spacing. TS0309 is the exact 75 ms ramp anchor.
Each shear phase lasts at least 75 ms and at least 30 ms beyond the nominal
ramp endpoint. The full-fault loading stop remains TS0278's one-`D_c`
coverage trigger.

TS0320 repeats TS0309's parameters but freezes loading at the first
`|V| >= 500 mm/s` in the loading-end 5-25 mm band. Its practically zero slip
threshold is intentional: normal loading already produces several `D_c` of
cumulative slip, so an absolute-slip trigger would not represent new nucleation.
No RSF state or cumulative slip is reset.

Each run stores 4400 bulk shear frames and 50000 interface shear frames. The
17-run combined estimated dump is about 1.26 TB and is gated below 1.4 TB
before Slurm launches the tasks. There is no animation on GB200.

## Diagnostics

`stats/rsf_rate_sweep_metrics.json` records the 200-469.5 mm front fits for
post-shear 0.05 and 1 `D_c`, the first 500 mm/s loading-end event, loading-stop
time/displacement, realized VW slip rates, and imposed-face work after that
event. The work is the trapezoidal integral of boundary reaction against the
prescribed displacement, in the model's energy units per unit out-of-plane
thickness. It is not inferred from the elastic-energy curve.

After the job, `runs/TS0304_TS0320_rsf_rate_sweep_summary.csv` gathers the
available per-run metrics and marks any missing analysis without discarding
completed simulation dumps.

Submit from `/work/gauss112/tatva` with
`sbatch slurm/PMMA-RSF-GB200-R1-RSF-RATE-SWEEP.slurm`.
