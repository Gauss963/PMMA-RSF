# TS0368-TS0383: short extension/timestep convergence campaign

Prepared 2026-09-29. Exactly 16 independent cases, one GB200 per case,
without MPI, animation, or animation frames. This implements the approved
`RSF_extension_short_convergence_plan.md`; submission and measured timing
are recorded below when available.

| Runs | Extension mm | dt ns in ascending run order |
| --- | ---: | --- |
| TS0368-TS0371 | 0 | 10, 5, 2.5, 1.25 |
| TS0372-TS0375 | 6 | 10, 5, 2.5, 1.25 |
| TS0376-TS0379 | 28 | 10, 5, 2.5, 1.25 |
| TS0380-TS0383 | 30 | 10, 5, 2.5, 1.25 |

## Unchanged physics

- Q4 mesh, h=0.5 mm, PMMA material and all RSF a/b/Dc profiles unchanged
  from TS0352-TS0367 (the original TS0343 friction/loading configuration).
- Full undamped explicit 40 ms normal phase, 20 ms ramp, 16 MPa applied
  normal traction. No prestress displacement or traction-consistent reset.
- Nominal half-cosine shear loading remains 0 to 2.45 mm in
  19.24225500323749 ms; stop at the first V>=500 mm/s in original y=5..25 mm
  with the same 1e-12 mm slip condition. No persistence/coverage change.
- Unloaded, noncontacting extension occupies -L_ext<=y<0. No normal load
  or friction acts there. Original fault remains y=0..500 mm.

Only the shear observation ends at 12 ms. A necessary schedule correction
permits observing a prefix of a longer nominal ramp without accelerating
it to the shorter observation interval. A regression compares this prefix
against the original long schedule exactly. A case that has not stopped
by 12 ms is explicitly flagged, not interpreted as a post-stop comparison.

## Common numerical corrections

All 16 cases, including the new 10 ns controls, use the same corrections:

1. Signed resisting traction uses the direction of the free relative
   velocity, not the projected velocity, which can underflow to zero while
   the impulse remains finite. This fixes saved traction diagnostics;
   the actual velocity projection is unchanged.
2. RSF theta storage and constant-rate ageing update use float64. Bulk
   displacement/velocity, FEM operators, and friction projection remain
   float32. Theta is cast to the bulk dtype for evaluating friction strength.
   This removes the lost at-rest dt increment in theta, not all possible
   float32 roundoff in displacement or cumulative slip.
3. Exact integer step indices define diagnostic time. The float32 running
   time is not used as the high-rate clock.

Manufactured ageing tests, tiny coupled runs, and interrupted/resumed runs
check these paths. A separate one-GPU GB200 pilot checks the actual backend
and times the full 0.5 mm / 30 mm-extension mesh at dt=1.25 ns. Its shortened
normal/shear phases are timing fixtures, not physical production results
or an additional TS case.

## Output and checkpoint

Each run uses `runs/TSxxxx/{input,data,stats,logs}`. Main dump:
`data/simulation.h5`.

- Bulk: 8 normal and 100 shear frames, displacement and stress only.
- Entire interface: 1000 normal and 120001 shear frames. The shear interval
  is at most 0.1 us for these timesteps.
- `integration_probes`: all 61 original interface nodes at y=0..30 mm,
  every integration step for the ENTIRE 12 ms shear phase. This deliberately
  supersedes the proposed narrow event windows: it also captures the late
  zero-extension peak without a fragile pre-trigger buffer.
- `values` stores eight float32 channels listed in `columns`: slip rate,
  total cumulative slip, signed plastic slip, mu, signed friction traction,
  normal overlap, normal traction, and actual post-constraint relative
  velocity. `state` stores float64 theta. `history` records the standard
  17 energy/loading/reaction channels at every shear step.
- Row i corresponds to shear step i+1; shear time=(i+1)*dt, absolute
  time=(normal_steps+i+1)*dt. Velocity is a staggered post-projection
  half-step value; displacement/normal overlap and state are full-step values.
- `saved_steps` indicates valid rows. Capacity is preallocated, so unread
  rows beyond this marker must not be treated as data after interruption.
- Host buffers are bounded to 4096 rows. Checkpoint creation flushes all
  pending diagnostics first. Resume checks dt, state dtype, coordinates,
  and dataset sizes, then safely overwrites rows after the checkpoint.
- Checkpoints every 10 wall-clock minutes and at phase ends, plus deadline
  and Slurm-signal handling. Runs start from t=0; no cross-dt restart.

Conservative raw array estimate: 0.265805 TB for all 16, about 8.30 to
29.43 GB/run. This is not an exact filesystem upper bound: chunk metadata,
checkpoints and compression overhead exist. Preflight additionally reserves
50 GB, and a running disk guard requests checkpointing below 25 GB free.
No compression saving is needed for the campaign to fit the 1.4 TB budget.

Streaming post-run summaries are stored in each run's
`stats/rsf_convergence_metrics.json`, with the campaign index at
`stats/TS0368_TS0383_convergence_summary.json`. They report exact stop time,
loading-end peaks with simultaneous contact stress, V*dt/Dc, slip integrals,
and separate post-stop shear/normal boundary work. A local peak velocity
is NOT a rupture-front velocity.

## Reproduction

```sh
python scripts/generate_ts0368_convergence_cases.py --check
python scripts/preflight_convergence_sweep.py
sbatch slurm/PMMA-RSF-GB200-CONVERGENCE-PILOT.slurm
# After a successful pilot, using its actual job ID/report:
sbatch --export=ALL,PILOT_REPORT=/work/gauss112/tatva/stats/convergence_pilot_JOBID.json \
  slurm/PMMA-RSF-GB200-R1-CONVERGENCE.slurm
```

The production launcher verifies the pilot solver/case fingerprint, exactly
16 cases, available storage, and one visible GPU per rank. The pilot adjusts
only its own pending dependent production job, using
`max(16, 2*ceil((1.25*predicted_finest_hours + 0.5)/2))` hours. Thus the
allowed limits are 16, 18, 20, 22, and 24 hours. A requirement above 24 hours
fails the pilot gate rather than silently starting an under-budget batch.
The initial 24-hour request is a conservative placeholder, not a measured
runtime or a guarantee that 16 hours is sufficient. The runner derives its
budget from the actual Slurm limit, minus one hour for checkpoint margin.
Completed physical cases exit rather than occupying their GPU with plotting.

## Submission record

### Current production submission (2026-09-29, 22:51 Taiwan time)

- **Job 457463**, `gb200-r1`, exactly TS0368-TS0383, 16 GPUs, no dependency.
- Solver/case fingerprint matches the successful numerical/timing portion
  of pilot 454906: `cb113eb923ad21fbfa7db36d9124457b5a9bfff9b0b9eb1cd101c3663b33a694`.
- The GPU mixed-precision/checkpoint tests PASSED. Measured full-mesh timing:
  normal 28.2132 s / 30720 steps; shear 16.0809 s / 12640 steps.
  Estimated finest-case duration is **11.5561 h**; adding 25% plus 0.5 h
  gives **14.9452 h**. These remain short-pilot extrapolations, not guarantees.
- Requested time limit: **16:00:00**. Runner checkpoints after at most
  15 hours, additionally capped to 30 minutes before the absolute cutoff.
- User deadline: **2026-10-01 00:00:00 Asia/Taipei**. Slurm scheduling
  deadline: **2026-09-30 23:50:00**, leaving ten minutes before the cutoff.
  With a full 16-hour reservation, latest eligible start is September 30
  at 07:50. Slurm may remove a job that can no longer finish in this window;
  neither queue availability nor completion of all physical cases is guaranteed.
- No additional per-rank pilot is run. The successful report fingerprint,
  case manifest, storage and one-GPU-per-task checks remain enabled.
- Pilot 454906's final scheduler-adjustment command was rejected by Slurm,
  so its batch status was FAILED even though both numerical validation and
  full-mesh timing had completed successfully. Dependent job 454914 was
  automatically cancelled without starting. Job 457463 supersedes it.
  Scheduler-update errors are now recorded separately and do not invalidate
  a successful numerical pilot.
- Production code at submission: `358ce72`.

### Earlier preparation

- Code: `1ff8143` (includes solver revision `6d517be` and backend checks).
- Timing/validation pilot: **454906**, `gb200-dev`, one GPU, 2-hour ceiling.
- Production: **454914**, `gb200-r1`, four nodes / 16 GPUs, exactly one
  independent TS case per GPU, dependency `afterok:454906`.
- At submission both are pending: dev nodes unavailable/reserved, production
  waiting for dependency. Production initially requests 24 hours, subject to
  the measured proportional adjustment above. No completed timing estimate
  is claimed yet.
- Pilot report: `/work/gauss112/tatva/stats/convergence_pilot_454906.json`.
- Production outputs: `/work/gauss112/tatva/runs/TS0368` through `TS0383`.
- Checkpoint interval: 10 minutes; Slurm warning: 30 minutes before timeout.
- Local regression: 44 tests passed; two additional scheduler-safety tests
  also passed. CPU normal- and shear-phase restarts agree bitwise.
- Earlier one-minute pilots 454886 and 454895 stopped on overly strict
  restart comparison assertions before the full-mesh timing phase. CUDA
  scatter/reduction roundoff is now compared separately per physical
  channel; the nearly cancelling signed mean shear traction uses the
  normal-contact stress scale for its absolute tolerance. Stop flags and
  actuator displacement still require exact equality. This changes the
  validation criterion, not the numerical solver. The earlier held
  production job 454903 was automatically cancelled on dependency failure
  and consumed no running allocation; it is not a second active sweep.

## Interpretation limits

This is a short temporal/precision diagnostic, not proof of full rupture
or spatial convergence. Compare event time AND event-relative waveforms,
matched-time energies, finite-width patch slip and traction impulses.
The finest step may still have V*dt/Dc too large during a sharp edge spike.
After inspecting this sweep, h=0.25 mm comparisons require a separate pilot
and approval; they are not part of these 16 cases.
