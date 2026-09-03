# PMMA Tatva case workflow

Each PMMA simulation starts from one TOML input and writes a numbered,
self-contained directory below `runs`.

```text
runs/TS0117/                          # first run in the new sequence
  ...
runs/TS0118/                          # local and Slurm use one sequence
  input/case.toml
  input/resolved_case.json
  input/command.txt
  data/simulation.h5
  checkpoint.npz                       # present only while resumable/incomplete
  stats/preflight.json
  stats/summary.json
  logs/job.log
  logs/nvidia_smi.csv
  status.json
```

Preflight a case without constructing a mesh or creating a run directory:

```bash
python scripts/run_case.py \
  cases/rsf_0124_q4_vs30_long_transition_12h.toml --preflight
```

Run it locally:

```bash
python scripts/run_case.py \
  cases/rsf_0124_q4_vs30_long_transition_12h.toml
```

Submit the default 0.50 mm Q4 explicit production case on one H200:

```bash
sbatch slurm/PMMA-RSF-GPU.slurm
```

The production driver currently executes one global JAX operator on one GPU.
An H200 scaling pilot took 268.995 s with one visible GPU and 270.888 s with
two visible GPUs (ratio 1.007); the second GPU remained idle. Therefore a
single production case must request one GPU until spatial domain decomposition
is implemented. Independent cases can still be assigned to separate GPUs.

The Slurm allocation is capped at twelve hours and requests one H200. Memory is
left unspecified so each cluster applies its automatic per-core allocation.
The runner checkpoints every ten minutes and exits cleanly
about 20 minutes before the allocation ends. Before creating the HDF5 dump it
requires enough actual filesystem space for the conservative uncompressed
estimate plus a 50 GB reserve. A checkpoint contains the complete explicit
integrator state and the exact HDF5 frame indices. Resume a cleanly
checkpointed run with:

```bash
sbatch --export=ALL,CASE_FILE=/work/gauss112/tatva/cases/rsf_0122_q4_slow_rsf_buffer_12h.toml,RESUME_DIR=/work/gauss112/tatva/runs/TS0122 \
  slurm/PMMA-RSF-GPU.slurm
```

The input is compared with `input/resolved_case.json` before resume. A mismatch
is rejected rather than silently mixing two cases. Completed runs remove the
checkpoint; interrupted runs keep both the flushed HDF5 and `checkpoint.npz`.

The PMMA entry point never imports plotting or animation scripts. Full bulk
fields and high-rate interface histories use independent frame counts so the
production case preserves fault timing without exceeding the dump budget.

An optional hybrid loading stage can construct the early shear preload with
damped dynamic relaxation before undamped explicit rupture dynamics:

```toml
[loading]
quasistatic_shear_fraction = 0.30
quasistatic_shear_start_time = 0.001
quasistatic_shear_ramp_time = 0.003
quasistatic_damping_time = 0.0002
```

The prescribed displacement uses the same half-cosine shape and RSF contact
projection as the explicit phase. Artificial damping is active only in the
normal/preload phase. At handoff, `stats/summary.json` records the kinetic to
stored-energy ratio, maximum slip, maximum slip rate, and actual displacement
under `summary.quasistatic_handoff`. A fraction must be piloted at the target
mesh: if dynamic slip begins before handoff, a static constraint would suppress
the intended RSF instability and the fraction is not physically admissible.
The 1 mm screening pilots first bracketed dynamic instability between 40% and
50%, but cumulative slip is the stricter handoff constraint. Target-mesh pilots
gave `max slip=0.0103 mm` at 20% and `0.0171 mm` at 30%, both above
`D_c=0.00630 mm`. Run 260651 showed why this matters: a 30% preload crossed the
old single-node stop trigger at the loading corner and froze the boundary at
the first explicit step. The corrected case uses a 10% preload, followed by a
9 ms explicit ramp so the active loading duration and velocity remain
unchanged. Loading now stops only when the same station has `slip >= D_c` and
`|slip rate| >= 10 mm/s` within `10 <= y <= 440 mm`. This excludes the measured
0-6 mm corner-slip zone while stopping external work shortly after a dynamic
front leaves the nucleation end.

The current TS0120 Q4 production case uses a 0.50 mm mesh, 1,443,584
displacement DOFs, an exact 10 ns step, 36,000 bulk shear frames, and 300,000
high-rate interface frames. It restores the 0116 normal-loading history as
undamped explicit dynamics: a 20 ms ramp followed by 20 ms at the prescribed
normal displacement, with friction active and no velocity reset at the shear
handoff. Only two normal bulk frames and 100 normal interface frames are saved.
Tangential loading starts from zero and follows a 2.45 mm half-cosine over the
full 29 ms shear phase.
TS0118 showed that a single y=440 mm station can be activated by an independent
far-edge disturbance before the loading-end rupture reaches it. TS0120 therefore
stops only after every interior station from y=0.5 to 499 mm has accumulated at
least D_c while a dynamic slip rate above 500 mm/s remains in that interval. If
that swept-front condition is never met, the half-cosine reaches zero velocity
naturally instead of freezing the loading face on a false local event.

The RSF parameters remain derived from the 5 mm cohesive-zone anchor in
`CohesiveZoneModel/Lc_estimate.py`, including shared
`D_c=0.000376505 mm`. The loading zone has `a=b=0.004` and is velocity
neutral. The conservative uncompressed estimate is 1.253 TB; the calibrated
LZF estimate is 1.196 TB and must remain below the configured 1.20 TB limit.

TS0121 keeps the TS0120 fully explicit 40 ms normal history and 2.45 mm shear
target, but lengthens the half-cosine shear ramp from 29 to 43 ms. The bulk and
interface shear targets increase to 53,379 and 444,828 frames, respectively,
so their physical sampling intervals remain approximately 0.806 and 0.0967
microseconds. The leading VS plateau is 40 mm long with `a-b=+0.006`; its 10 mm
half-cosine transition begins at y=450 mm and covers the TS0120 reverse nucleus
near y=458 mm. TS0121 omits bulk strain from HDF5 because no production
post-processor consumes it; displacement, velocity, stress, and all interface
fields remain complete. This lowers the calibrated dump estimate to 1.187 TB
without changing the solver or stress calculation.

TS0122 replaces the hard TS0121 terminal barrier with a standard RSF buffer.
The last 50 mm has `f0=0.70` and `a-b=+0.008`, joined to the middle fault by a
50 mm half-cosine transition. Its lower low-speed strength permits stable
prestress release, while velocity strengthening raises its steady friction to
approximately 0.834 at 2000 mm/s. The 2.45 mm half-cosine ramp is lengthened
to 57 ms; 70,759 bulk shear frames and 589,655 interface frames preserve the
TS0120 physical sampling intervals. Bulk velocity is omitted together with
bulk strain because production analysis uses the stored displacement, stress,
scalar energy history, and high-rate interface slip rate. The calibrated dump
estimate remains 1.183 TB.

TS0123 is a controlled ablation of the TS0122 terminal buffer. The leading
zone is identical to the middle velocity-weakening zone, so there is no
terminal `f0`, `a`, `b`, or `D_c` gradient. The neutral 30 mm loading zone and
its 50 mm transition are retained. The 2.45 mm half-cosine shear ramp is
lengthened to 75 ms, reducing peak boundary velocity while leaving the fully
explicit 40 ms normal phase and 10 ns step unchanged. The dump budget is 1.40
TB; bulk velocity and strain remain omitted while stress, displacement,
energies, and high-rate interface histories are retained. The configured
83,200 bulk and 800,000 interface shear frames give a calibrated estimate of
approximately 1.394 TB.

TS0124 restores a moderate terminal velocity-strengthening zone after the
uniform TS0123 ablation increased reverse rupture. The final 30 mm plateau uses
`f0=0.8`, `a=0.008`, and `b=0.005`. Its leading-edge half-cosine transition is
lengthened to 100 mm, from y=370 to 470 mm, while the loading-end transition
remains fixed at 50 mm. Separate loading and leading transition lengths avoid
changing nucleation while testing the smoother terminal profile. Loading,
time step, frame counts, and the 1.40 TB dump budget are unchanged from
TS0123.

TS0125 is a controlled near-velocity-neutral test of the TS0124 terminal zone.
The geometry, loading history, 30 mm plateau, 100 mm leading transition,
time step, frame counts, and dump budget are unchanged. Only the leading
direct effect decreases from `a=0.008` to `a=0.006`, while `b=0.005` and
`f0=0.8` remain fixed. The resulting `a-b=+0.001` tests whether reducing the
terminal velocity-strengthening impedance promotes stable prestress release
without introducing a low-strength patch.

TS0126 moves the planned geometry test ahead of the exact velocity-neutral
test after TS0125 produced no meaningful improvement. It restores the complete
TS0124 RSF profile and removes a 20 mm by 5 mm triangular wedge from the moving
block's leading corner. A continuous coordinate mapping preserves the all-Q4
topology and avoids a staircase boundary. The active contact ends at y=480 mm,
so rupture coverage is evaluated through y=479 mm. The RSF profile remains
anchored to the original 500 mm coordinates; all surviving fault stations
therefore retain exactly the TS0124 `f0`, `a`, `b`, and `D_c` values. Loading,
time step, output rates, and the 1.40 TB dump budget remain unchanged.

TS0144 through TS0159 form a controlled shear-loading-rate sweep based on
TS0126. Geometry, the VN/VW/VS profile, normal loading, 2.45 mm target shear
displacement, 75 ms shear window, 10 ns step, and rupture-stop rule are held
fixed. The half-cosine peak-speed multiplier is linearly spaced from 1 to 3 in
16 values, so the ramp duration is `0.075 / multiplier` seconds and ranges
from 75 to 25 ms. Each case retains 1/16 of the TS0126 bulk and interface
shear-frame targets, giving an estimated 0.0871 TB per run and 1.394 TB for
the complete sweep. The committed cases are generated and verified by
`scripts/generate_shear_rate_sweep_cases.py`; the GB200-r1 launcher assigns
one independent case to each of 16 GPUs without MPI.
