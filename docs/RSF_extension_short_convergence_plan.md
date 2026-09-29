# Proposed short convergence checks after TS0352-TS0367

Original proposal, 2026-09-29. Subsequently approved for implementation.
The exact TS0368-TS0383 cases, implemented diagnostics, precision corrections
and submission record are in `TS0368_TS0383_timestep_convergence.md`.
The sections below retain the original planning assumptions for traceability.

## Scope

Determine whether the loading-end slip-rate spikes and the nonmonotonic
28 versus 30 mm response survive numerical refinement. This is not yet a
test of full rupture over the original 95 ms shear observation interval.
Keep the original friction law, a/b/Dc, boundary conditions, contact model,
normal loading, nominal shear ramp and first-500-mm/s stop rule unchanged.
Do not change modulus or increase the imposed displacement in this test.

## Preliminary checks

Before using a large allocation, repair and test the already identified
near-sticking signed-traction diagnostic. Compare reported traction against
the actual momentum correction; an underflowed zero slip velocity must not
erase a finite resisting traction. Do not rewrite historical dumps.

Check floating-point accuracy separately from timestep accuracy. At this
case's initial theta=4.2852936 s, float32 spacing is about 4.77e-7 s: adding
10 ns to theta rounds back to theta. The existing ageing update can thus
lose the at-rest age increment. Its ideal 40 ms contribution changes the
state-dependent coefficient by only about 0.000271, not evidence that it
explains the large rupture difference. Nevertheless, reducing dt alone
does not remove a precision floor. Use manufactured constant-rate tests
and a small float64 reference before interpreting a dt trend. These
diagnostic/precision changes have not been implemented by this proposal.

## First production diagnostic: time refinement

Exactly 16 cases if using the established 16-GPU allocation:

| Extension length mm | Reference case | h mm | dt ns |
| ---: | --- | ---: | --- |
| 0 | TS0352 | 0.5 | 10, 5, 2.5, 1.25 |
| 6 | TS0355 | 0.5 | 10, 5, 2.5, 1.25 |
| 28 | TS0366 | 0.5 | 10, 5, 2.5, 1.25 |
| 30 | TS0367 | 0.5 | 10, 5, 2.5, 1.25 |

One independent run per GPU, no MPI. Rerun the 10 ns controls with the same
diagnostic output as the refined cases rather than comparing narrow spikes
sampled at different cadences.

Retain 40 ms of undamped explicit normal loading (20 ms ramp). Tentatively
stop the shear observation at 12 ms, instead of 95 ms, while leaving the
nominal 19.242255 ms displacement ramp unchanged. The existing cases latch
the actual actuator stop at about 5.1-7.7 ms. Before finalizing the 12 ms
cutoff, check the peak times in the F1 supplementary diagnostics: extend the
window if needed to retain the relevant spike and at least 2 ms thereafter.
Flag any refined run that does not latch its stop before this cutoff; do
not silently treat a still-loaded case as a comparable post-stop result.

F1 job 1102618 subsequently completed. The four selected cases' saved
global-peak times are 7.57158, 7.28753, 7.70838 and 7.33978 ms after shear
starts, respectively. Thus 12 ms includes these existing peaks. The 6, 28
and 30 mm peaks occur 10.45, 19.00 and 21.85 us after actuator arrest;
the zero-extension peak is 2.44532 ms after arrest. Include a separate
window around that later zero-extension peak, not just the stop window.
Measure wave travel and contact response around stopping: these short delays
motivate checking stop-generated transients but do not prove their cause.
At the 6 and 28 mm peaks, saved strength/coefficient imply local compression
of approximately 150 and 147 MPa, much larger than the 16 MPa applied load.
Direct gap/penetration/normal-traction diagnostics and spatial refinement of
this contact edge are therefore important, not only a smaller dt.

Start each variant from t=0. Current resume validates dt, step counts and
frame counts; it cannot change dt or mesh, and completed checkpoints are
removed. A future branch-restart facility would also need to correctly
rephase the staggered velocity and demonstrate equivalence to a from-zero
run. Copying a displacement frame is not a valid restart of RSF dynamics.

Using the observed 3.63-4.34 h per original 135 ms run, the crude step-count
estimate for 52 ms is:

| dt ns | Estimated wall time per GPU |
| ---: | ---: |
| 10 | 1.4-1.7 h |
| 5 | 2.8-3.3 h |
| 2.5 | 5.6-6.7 h |
| 1.25 | 11.2-13.4 h |

These are estimates, not benchmarks or completion guarantees. Compilation,
event-local output, precision changes and different solver behavior can
change cost. A timing pilot is required before setting the production limit.

## Diagnostics and data volume

Do not dump bulk stress or render animation at every integration step.
Suggested outputs, requiring an event-window diagnostic writer:

- Entire fault: interval no larger than 0.1 us during the short shear phase.
- Original loading-end y=0..30 mm: every integration step in a narrow event
  window, with a pre-trigger buffer to capture the triggering spike. An
  initial window is 0.1 ms before to 0.5 ms after the event; adjust it from
  actual peak times, and keep a cheap running maximum throughout the run.
- Capture slip rate, cumulative slip, theta, mu, signed impulse-derived
  traction, contact compression/gap and actuator reaction/stop information.
- Record energy/work histories at high rate; retain only roughly 100 bulk
  frames per short run plus sparse normal-phase frames. No animation.

For 1001 interface stations, seven float32 interface fields, and 12 ms at
0.1 us cadence, raw interface arrays occupy about 3.36 GB/run. The 61-node
0.6 ms every-step window occupies about 0.82 GB at dt=1.25 ns. Sixteen runs
with sparse bulk output are therefore approximately a hundred-GB-scale
campaign, not a TB-scale one, before added channels, chunk overhead and
compression. This is a proposed output layout, not a current input option.

## What counts as convergence

Compare fixed physical positions/patches, not only whichever node has the
largest peak. Use both absolute shear-phase time and time relative to actuator
stop; event alignment alone could conceal a large nucleation-time error.

1. Absolute first-500 event time, stop displacement and pre-stop work.
2. Local waveforms, pulse width, integrated slip, state change and traction
   impulse in y=0..1, 5..25, and 30..150 mm. Retain the sharp peak without
   smoothing; also compare finite-width patch averages across meshes.
3. Connected moving-front arrivals at multiple thresholds, central slip and
   propagation/arrest extent. Do not infer speed from isolated peaks.
4. Matched-time extension/bulk/interface/kinetic energies and both normal and
   shear boundary work. Do not combine independently timed extrema.

Suggested acceptance targets, not universal constants: finest two dt levels
within about 5% for integrated slip, work and energy measures, and about 1%
for stop time/displacement, without changes in propagation/arrest behavior.
Require a decreasing refinement trend, not merely one pair that agrees.
If peak amplitude does not converge while patch-integrated quantities do,
investigate the sharp contact-edge stress concentration rather than labeling
the divergent point peak a physical rupture velocity.

Dc=0.428529 um. The largest saved peak in TS0366 gives V*dt/Dc~2.90 at 10 ns.
Holding V fixed would give 1.45, 0.725, 0.363 at the finer steps. Actual V can
change and saved output may miss still larger peaks, so 1.25 ns is not
guaranteed sufficient. An exact constant-V state update preserves positivity
but does not establish accuracy of the coupled contact/dynamic solution.

Lapusta et al. (2000), section 4, explicitly motivate a time-increment
criterion using slip per step relative to Dc. Their spectral solver's
specific stability constants cannot be transferred directly to this FEM
implementation. The relevant lesson is to test constitutive time resolution
as well as elastic-wave CFL stability:
https://www.its.caltech.edu/~lapusta/Lapusta_et_al_JGR2000.pdf

## Second stage: spatial refinement

Only after the temporal/precision check, select one or two informative
extensions (likely 6 and 28 mm). Compare h=0.5 and 0.25 mm at the SAME dt
shown adequate on both meshes; add an intermediate or finer mesh if the
two-level comparison is not stable. Hold physical loading/contact parameters
fixed and audit any automatic mesh-dependent penalty settings.

Halving h in 2-D increases element count by approximately four; if dt must
also be halved, cost is approximately eight times the coarse counterpart.
Fine-mesh runtimes must therefore be piloted separately rather than assuming
the temporal-sweep budget covers them. A two-mesh agreement is an initial
check, not a formal asymptotic convergence-order estimate.

Do not interpret the first short campaign as approval to change physical
loading or as proof that late full-fault rupture has become possible.
