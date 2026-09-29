# TS0352-TS0367: unloaded loading-extension results

Reviewed 2026-09-29 from completed dumps transferred to F1. No simulation
parameters or historical dumps were changed during this review.

## Conclusion

The extension helps local slip and increases the stored energy available
after actuator arrest, but **none of the 16 cases demonstrates the requested
clean, full, fast forward rupture**. The benefit is strongly nonmonotonic.
TS0366 (28 mm) has the largest central median slip and longest connected
high-peak-rate region, but it is not a validated successful rupture case.

All 16 have zero measured post-stop SHEAR actuator work. Slow evolution still
occurs after stopping. Continued shear pushing therefore is not necessary for
the slow slip seen in this sweep. This does not prove energy is sufficient for
fast rupture: spatial stress availability, nucleation maturity and dissipation
still matter. Normal traction remains applied and can do nonzero work.

## Jobs and outputs

- F1 all-standard-plots job: `1102579`, 32 allocated CPUs, four concurrent runs,
  using `slurm/PMMA-ANALYSIS-SWEEP-CPU.slurm`. No animations or animation frames.
- F1 bounded-memory comparison/diagnostic job: `1102618`, four CPUs,
  using `slurm/PMMA-EXTENSION-REVIEW-CPU.slurm`.
- Individual figures: `/work1/gauss112/tatva/runs/TS0352/Plot` through
  `/work1/gauss112/tatva/runs/TS0367/Plot`.
- Comparison PDF/PNG: `/work1/gauss112/tatva/reviews/TS0352_TS0367/Plot`.
- Comparison CSV/JSON: `/work1/gauss112/tatva/reviews/TS0352_TS0367/stats`.
- Per-run logs: `runs/TS*/logs/analysis-1102579.log`; consult postprocessing
  status for unavailable/failed individual analyses rather than interpreting
  a successful Slurm exit as proof every scientific fit was applicable.

The all-plots job was running at review time, not yet complete. The comparison
figures were rendered and visually checked locally from the transferred
metrics and station tables. The supplementary diagnostic job reads scalar
histories and one high-rate station trace per run, not full bulk stress arrays.

## Common experiment and control

L_ext = 0, 2, ..., 30 mm. The extension is continuous PMMA below original y=0,
has no contact and no external normal traction. The 500 mm fault and its RSF
profile remain fixed. The displacement boundary moves to y=-L_ext.

Normal stress is 16 MPa; normal phase is 40 ms with 20 ms ramp. Nominal shear
half-cosine ramp is 2.45 mm / 19.242255 ms; it stops immediately at the first
500 mm/s monitored slip-rate event in original y=5..25 mm. The actual ramp is
therefore much shorter than nominal. Total shear duration is 95 ms.

Q4 h=0.5 mm, dt=10 ns, float32. Dc=0.00042852936846 mm (0.428529 um).
Leading y=470..500 mm is weakly VS. No artificial state handoff or added
normal-phase shear prestress. Use **TS0352 as the zero-extension control**:
old TS0343 used a different near-sticking projection solver. All current cases
share the corrected log-speed projection.

## All 16 cases

Energy units are N mm per mm out-of-plane thickness. The release column is
U_ext(stop) minus its minimum after stopping; different energy minima need
not occur at the same time. Middle slip is the median accumulated slip since
normal end over 200 <= y < 470 mm.

The last column is the end of the longest connected spatial region that
intersects y=5..25 mm and whose stations each reached V>=500 mm/s at some
saved time. It is **not** a time-coherent front, rupture speed or instantaneous
arrest position. Isolated remote peaks are excluded, without filling gaps.

| Run | L_ext mm | Stop displacement mm | U_ext at stop | Max release | Middle slip um | Peak-rate region end mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TS0352 | 0 | 0.4045 | 0.00 | 0.00 | 0.365 | 5.0 |
| TS0353 | 2 | 0.6514 | 11.96 | 4.02 | 0.477 | 67.0 |
| TS0354 | 4 | 0.6450 | 20.81 | 6.36 | 0.883 | 64.0 |
| TS0355 | 6 | 0.7675 | 33.33 | 12.97 | 15.299 | 118.5 |
| TS0356 | 8 | 0.7307 | 38.97 | 13.71 | 1.881 | 106.0 |
| TS0357 | 10 | 0.7364 | 46.00 | 15.49 | 2.749 | 109.0 |
| TS0358 | 12 | 0.7486 | 53.63 | 18.36 | 2.161 | 108.0 |
| TS0359 | 14 | 0.7566 | 60.02 | 21.28 | 7.915 | 111.5 |
| TS0360 | 16 | 0.7825 | 67.99 | 24.47 | 3.731 | 129.5 |
| TS0361 | 18 | 0.8083 | 76.10 | 28.09 | 13.703 | 136.5 |
| TS0362 | 20 | 0.7639 | 76.73 | 26.49 | 4.894 | 109.5 |
| TS0363 | 22 | 0.7597 | 81.01 | 30.70 | 12.522 | 107.0 |
| TS0364 | 24 | 0.7626 | 86.00 | 27.52 | 6.897 | 111.0 |
| TS0365 | 26 | 0.7571 | 89.54 | 27.55 | 1.601 | 107.5 |
| TS0366 | 28 | 0.8448 | 106.75 | 40.02 | 23.694 | 152.0 |
| TS0367 | 30 | 0.7751 | 101.08 | 33.04 | 3.538 | 114.5 |

## Why this is improvement but not full rupture

1. Central slip increases substantially in several extensions (e.g. 0.365 to
   23.694 um for 0 to 28 mm), and extension energy is demonstrably released.
   This supports the compliance/energy-reservoir idea locally.
2. The high-slip-rate activity remains concentrated near the loading end.
   In 200 <= y < 470 mm, only TS0360 and TS0366 contain a single saved station
   reaching 500 mm/s; the other cases contain none. No case has adequate
   central coverage for a 500/1000 mm/s arrival fit. This threshold is an
   operational diagnostic, not a universal physical definition of rupture.
3. Even TS0366 has only 49.2% of leading-region stations accumulate one Dc;
   the y=500 endpoint remains effectively unslipped in every case. This is
   not merely a single endpoint problem: a finite part of the tail stays
   below Dc. Whole-interface >=Dc coverage of 96.9% is not full rupture.
4. The original 1Dc-contour speeds in positive-extension cases are roughly
   5.9-24.7 m/s, with variable fit quality. They are slow weakening contours,
   not independently established dynamic crack-tip speeds. Actual mu-map PNGs
   inspected for TS0352 and TS0355 are consistent with delayed, broad evolution.

In particular, TS0352's map currently labels a negative speed with R^2~0.035.
That fit is not evidence for a reverse rupture. It is a regression through
incoherent 1Dc crossings and should not be used scientifically. Likewise,
black in the current mu maps means coefficient <0.6, not "ruptured". An RSF
interface can support low shear traction with exponentially small velocity.
The two normal-phase bulk frames do not resolve normal-phase evolution; the
high-rate interface data, rather than those two frames, support this review.

## Two confounders prevent an energy-only interpretation

**The zero-length boundary condition changes locally.** With L_ext=0, original
y=0 is on the imposed-displacement face; for L_ext>0 it is no longer directly
constrained. The maximum normal-phase accumulated slip falls from 1.975 um in
TS0352 to numerical near-zero in the positive extensions. This changes the
loading-end state before shear loading, not just the amount of extra PMMA.

**The stop rule allows different pre-stop loading.** Stop displacement rises
from 0.4045 mm to 0.6450-0.8448 mm. Integrated shear-boundary work rises from
542.72 to approximately 1074-1568 model energy units. Extension-length effects
therefore include different nucleation times and received work. Identical
nominal displacement programs are not identical pre-rupture stress states.

In TS0366, the extension holds 106.75 units at stopping and releases at most
40.02. Total elastic-plus-interface energy falls by at most 246.93, while
normal-boundary work over the entire post-stop interval is +179.83. These
numbers refer to different extrema/intervals and must not be summed as an
energy balance. The diagnostic JSON also computes normal work and extension
release at the same total-energy minimum. Zero post-stop shear work does
not imply a mechanically isolated specimen.

## Numerical and diagnostic limitations

### Very localized fast slip needs convergence checks

Positive extensions have global saved slip-rate peaks of about 26-124 m/s,
localized around y=0..1 mm. These are particle-relative slip velocities, not
rupture propagation velocities. With the prescribed Dc and dt,

    max(V_saved * dt / Dc) = 0.61..2.90.

Ten of the fifteen positive extensions exceed one Dc per integration step
at a saved peak. This is a local temporal-accuracy warning, not by itself proof
of solver instability. An exact constant-velocity state update does not ensure
the coupled acceleration/contact problem is accurately resolved at that rate.
The 0.95 us interface output cadence is much longer than the 10 ns step, so
unsaved peaks can be larger and burst chronology can be unresolved.

Do not call TS0366 the best physical solution just because its peak and slip
are largest. A short high-rate-output dt/mesh convergence test around the
loading-end event is needed before choosing an extension for another sweep.

### Near-sticking signed-traction output still needs repair

Code inspection found a separate diagnostic issue in
`tatva/pmma/dynamics.py`, `apply_regularized_friction`: the magnitude returned
by the log-speed solve is multiplied by `sign(corrected_velocity)`. If the
physical velocity underflows float32 to zero, this erases finite traction
from saved `friction_strength` and the signed average-traction diagnostic.
The directly saved coefficient still uses the finite log-space strength.
The momentum update uses the velocity correction instead, so this observation
does not establish a force-update error or explain the whole rupture outcome.

The diagnostic script counts zero saved strength with nonzero coefficient at
arrest. Historical signed sticking traction cannot simply be reconstructed
from unsigned mu and overwritten. No old dump was rewritten, and this review
does not use that traction channel as a reliable near-sticking prestress.
The production diagnostic repair and its regression test remain a follow-up,
not an unreported change to this completed campaign.

## Recommended interpretation and next decision

Keep the extension idea as a useful change in boundary compliance, but the
0-30 mm experiment does not yet support the claim that it produces autonomous
high-speed rupture. Increasing length blindly is not justified by the
nonmonotonic results. Before another large campaign:

1. Resolve the local peak's timestep/spatial convergence and repair the
   near-sticking diagnostic channel; compare 0, 6 and 28 mm controls.
2. Compare actual stress, age, incremental stored energy and connected slipping
   patch at stopping. Separate the new y=0 constraint from extension storage.
3. If insufficient mature nucleation is confirmed, discuss a connected,
   persistent nucleation stop criterion or a physically motivated stress
   preparation. Changing the literal first-500 rule needs user agreement.

No new simulation campaign was launched as part of this analysis request.
