# TS0336-TS0351: results and diagnostic audit

Reviewed 2026-09-26 from `/work1/gauss112/tatva/runs/` on F1.
No simulation settings were changed and no new jobs were submitted during this review.

## Conclusion

The continuous preparation and immediate shear-displacement stop worked, but
none of the 16 cases demonstrates the requested clean, self-sustaining,
loading-end-to-leading-end dynamic rupture. This is not evidence that the
remaining problem is simply an incorrectly chosen direct-effect parameter.

Two issues must be separated:

1. Stopping at the first single-node 500 mm/s crossing leaves substantially
   less shear loading than TS0319. The resulting local events do not develop
   into a coherent dynamic front across the central fault.
2. A reproducible low-speed RSF projection error contaminates reported
   friction strength and coefficient. The different background colors in
   the mu maps are partly a numerical diagnostic artifact, not a rupture map.

## Evidence and scope

- Read every run's `rsf_nucleation_sweep_metrics.json`, station-arrival CSV,
  `rsf_rupture_analysis_metrics.json`, summary and postprocessing status.
- Inspected the actual phase-split PNGs for TS0336, TS0343 and TS0347.
- Read selected high-rate HDF5 histories and stop-event windows for TS0336,
  TS0343, TS0347 and TS0350. No bulk stress volumes were loaded.
- Compared with retained TS0319 metrics. Its original HDF5 was no longer at
  the F1 path, so the old result was not independently re-extracted.
- Time estimates below use integer step IDs and dt, not accumulated float32 time.

All 16 cases have `state_reinitialized_at_handoff=false`. The maximum absolute
log state change between the saved frames bracketing the phase boundary is
1.85e-6. All 16 have zero measured post-stop SHEAR boundary work. Constant
normal traction remains applied; zero shear work does not imply zero work
from every external boundary.

In y = 200 to <470 mm, no saved station in any case reaches 500 mm/s. The
maximum saved central slip rates are 315-450 mm/s. These are slip velocities,
not rupture-propagation velocities. No sustained 500/1000 mm/s arrival front
can be fitted there. Lowering the diagnostic threshold to 100 mm/s still gives
nonmonotonic arrivals and poor linear fits (R-squared at most about 0.143).
This threshold check alone is not a universal definition of dynamic rupture;
the spatially incoherent arrivals and small slips are additional evidence.

### All cases

Middle-slip medians are incremental accumulated slip since the end of normal
loading, over y = 200 to <470 mm. The >=Dc fraction is the existing metric
over all saved contact stations, including constrained end stations. It is
NOT a full-rupture fraction or a connected-front criterion.

| Run | a | Nominal ramp ms | Stop ms after shear starts | Stop displacement mm | Middle median slip um | Stations >=Dc % | Peak central V mm/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TS0336 | 0.0025 | 75.000 | 28.079 | 0.7540 | 11.981 | 95.9 | 420.7 |
| TS0337 | 0.0025 | 50.000 | 12.107 | 0.3376 | 0.797 | 89.3 | 415.8 |
| TS0338 | 0.0025 | 30.000 | 8.197 | 0.4242 | 0.247 | 43.5 | 315.1 |
| TS0339 | 0.0025 | 19.242 | 4.977 | 0.3827 | 0.801 | 88.0 | 407.7 |
| TS0340 | 0.0050 | 75.000 | 21.487 | 0.4636 | 0.182 | 40.2 | 341.4 |
| TS0341 | 0.0050 | 50.000 | 14.011 | 0.4448 | 0.195 | 45.8 | 342.8 |
| TS0342 | 0.0050 | 30.000 | 7.880 | 0.3939 | 0.684 | 82.3 | 413.0 |
| TS0343 | 0.0050 | 19.242 | 4.912 | 0.3731 | 0.978 | 95.1 | 409.1 |
| TS0344 | 0.0100 | 75.000 | 21.091 | 0.4478 | 0.511 | 79.6 | 420.6 |
| TS0345 | 0.0100 | 50.000 | 14.161 | 0.4537 | 0.146 | 40.1 | 345.0 |
| TS0346 | 0.0100 | 30.000 | 7.733 | 0.3802 | 0.417 | 71.8 | 363.9 |
| TS0347 | 0.0100 | 19.242 | 4.811 | 0.3588 | 0.757 | 92.1 | 396.4 |
| TS0348 | 0.0200 | 75.000 | 20.177 | 0.4121 | 0.121 | 47.7 | 361.2 |
| TS0349 | 0.0200 | 50.000 | 12.614 | 0.3650 | 0.456 | 83.7 | 415.3 |
| TS0350 | 0.0200 | 30.000 | 8.123 | 0.4170 | 0.146 | 51.2 | 442.6 |
| TS0351 | 0.0200 | 19.242 | 4.717 | 0.3456 | 0.597 | 86.6 | 449.9 |

TS0336 has the largest central slip, but the broad slow evolution is not the
desired fast, clean front. Its 1Dc-contour fit is 14.0 m/s with R-squared 0.584;
this is not a defensible steady dynamic rupture speed. Across the entire
sweep, 1Dc fits have R-squared 0.004-0.584, and several slopes are negative.
A negative regression through disconnected first crossings does not by
itself establish a reverse-propagating rupture.

## Why the early stop matters

TS0343 retains the TS0319 a, b, Dc, nominal ramp and displacement amplitude.
TS0319 already had steady-state initial age, rather than the later
traction-consistent handoff. The principal loading change is the stop rule;
saved cadence and the total hold duration also differ.

| Quantity | TS0319 | TS0343 |
| --- | ---: | ---: |
| Nominal ramp ms | 19.242 | 19.242 |
| Actual stop ms | 11.012 | 4.912 |
| Actual stop displacement mm | 1.5007 | 0.3731 |
| Integrated shear-boundary work, model units | 3947.13 | 484.13 |
| Peak central slip velocity mm/s | 18544.97 | 409.12 |

TS0343 receives 12.27% of the old total shear work and stops at 24.86% of the
old displacement. TS0319 received 82.67% of its total shear work after its
first saved loading-end 500 mm/s event. It therefore does not demonstrate
that its rupture could have propagated without continued loading.

The new stop checks every 10 ns; high-rate output is every 0.95 us. Five cases
have no saved 500 mm/s crossing within the monitored y = 5-25 mm interval,
despite a latched stop. A shorter unsaved spike is possible: this is not proof
that the stop code malfunctioned or that every trigger was numerical noise.

Where a trigger was resolved, it was local. In TS0347, only one station was
above 500 mm/s in the first stopped frame, with at most four monitored stations
above it in the next few saved frames. In TS0350 a local burst reached about
4.98 m/s, but never established a central dynamic front. A single velocity
crossing is consequently not a validated test for mature, autonomous nucleation.

Post-stop peak kinetic energy is 0.142-0.173% of total stored elastic plus
interface energy at stopping. This denominator includes the large normal-load
energy. For TS0343, dividing instead by the stored-energy increase since the
end of normal loading gives about 2.82%. Neither ratio, alone, establishes a
successful rupture or isolates a purely shear energy budget.

Normal loading has small localized accumulated slip (global maxima about
1.6-2.4 um), but no saved 500 mm/s event; the central median normal slip is
effectively zero. Do not infer normal-phase rupture from a black mu-map region.

## Reproduced low-speed projection defect

`tatva/friction.py:174`, `project_regularized_rate_state_velocity`, bisects
the LINEAR velocity interval [0, V_free] 64 times and then returns the midpoint
and friction strength, without checking the residual. Its intended equation is

    V + A * tau(V, theta) = V_free.

For a nearly locked node the true RSF root can be exponentially smaller than
V_free / 2^65. The returned midpoint then has the wrong strength even though
its velocity is practically zero.

Reproduction using current code: sigma=16 MPa, f0=0.8, V0=1e-4 mm/s,
theta=Dc/V0, b-a=0.024123527228205266, preserved b*Dc, and
A=69.68641115 (the nominal two-sided 0.5 mm Q4 interface impulse factor).
Set V_free=A*16*0.1, so a nearly sticking solution requires tau/sigma=0.1.

| a | Returned mu, 64 iterations | Mu from the applied momentum correction |
| --- | ---: | ---: |
| 0.0025 | 0.722174 | 0.100000 |
| 0.0050 | 0.644349 | 0.100000 |
| 0.0100 | 0.488698 | 0.100000 |
| 0.0200 | 0.177396 | 0.100000 |

Returned V is about 3.022e-18 mm/s. Float32 and float64 give essentially the
same erroneous reported strengths when both use 64 linear bisections.
Changing dtype alone is not a solution to the bracketing-resolution problem.

This pattern matches the normal-end map backgrounds in the actual dumps:
approximately 0.72, 0.64, 0.49 and 0.17 as a increases. Since the plot masks
mu <0.6 in black, the larger-a runs can appear already weakened at rest.
This is not evidence that those stations have dynamically ruptured.

In the PMMA stepping code, velocity is updated from the velocity correction,
whereas this returned strength is used in reported interface traction and mu.
The reproduction therefore proves a diagnostic/implicit-residual inconsistency;
it does NOT prove that this defect alone caused all the arrested events.
Bulk stress, impulse traction and state evolution need an explicit consistency
audit before interpreting mu maps or fitted CZM parameters from this regime.
Successful TPV101/102 comparisons do not exercise every near-zero-velocity,
small-a PMMA condition and do not rule out this defect.

## Interpretation and next step

Keep continuous preparation and avoid the artificial traction-consistent
handoff. Do not choose a winner from map color or the present 1Dc regressions.
There is no systematic success trend with either ramp duration or a.

First repair and regression-test the low-speed implicit solve and the
consistency between impulse traction, returned traction, mu and state. Include
near-sticking PMMA cases in addition to the existing TPV tests. Mask invalid
speed fits, and make plots without full rupture explicitly say that a
post-full-rupture baseline is unavailable.

Then use a short diagnostic pilot with integration-step output around the stop
event. If the stop rule is revised, use a connected accelerating patch and a
temporal persistence test, not only a larger single-point threshold. This would
change the user's literal first-500 rule and needs agreement before a new
campaign. If first-500 must remain exact, a different physically prepared
stress distribution or nucleation setup must make that event autonomous; merely
sweeping a and nominal loading rate did not accomplish that here.

For context, RSF nucleation depends on patch stiffness/size and state as well
as velocity. With nominal sigma=16 MPa, the dimensional scale
G' Dc/[sigma(b-a)] is about 2.9-4.8 mm in this sweep. This is only a scale,
not an exact boundary-nucleation length for the finite blocks. The homogeneous
half-space ageing-law results also depend on a/b and V*theta/Dc.
See [Rubin and Ampuero (2005)](https://doi.org/10.1029/2005JB003686) and
[Ampuero and Rubin (2008)](https://doi.org/10.1029/2007JB005082).

## Postprocessing limitations

All 16 postprocessing runs wrote final status reports. Three bundles are
complete; thirteen are `complete_with_failures`, as intended by the
continue-on-plot-error policy. The logged errors are 91 late-plateau requests
preceding full-fault rupture and five frame-time bounds outside the saved
phase. These are analysis-window assumptions, not logged OOM failures.
The phase-split, history and RSF profile outputs are available. Missing
late-plateau figures should not be mistaken for proof of full rupture.

The normal-phase panel is based on only two bulk samples. Center-to-edge
pcolormesh extrapolation can display a -20 to 60 ms extent for a real 0-40 ms
phase; this visual artifact does not mean the normal loading actually lasted
60 ms. The high-rate normal history contains 1000 samples and was used for
the quantitative checks above.
