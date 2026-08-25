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
  cases/rsf_0118_q4_explicit_10h.toml --preflight
```

Run it locally:

```bash
python scripts/run_case.py \
  cases/rsf_0118_q4_explicit_10h.toml
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

The Slurm allocation is capped at ten hours and requests one H200. Memory is
left unspecified so each cluster applies its automatic per-core allocation.
The runner checkpoints every ten minutes and exits cleanly
about 20 minutes before the allocation ends. Before creating the HDF5 dump it
requires enough actual filesystem space for the conservative uncompressed
estimate plus a 50 GB reserve. A checkpoint contains the complete explicit
integrator state and the exact HDF5 frame indices. Resume a cleanly
checkpointed run with:

```bash
sbatch --export=ALL,CASE_FILE=/work/gauss112/tatva/cases/rsf_0118_q4_explicit_10h.toml,RESUME_DIR=/work/gauss112/tatva/runs/TS0118 \
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

The current TS0118 Q4 production case uses a 0.50 mm mesh, 1,443,584
displacement DOFs, an exact 10 ns step, 36,000 bulk shear frames, and 300,000
high-rate interface frames. The 2.22 mm half-cosine loading is spread over the
full 29 ms shear phase. A 1 mm pilot reached dynamic rupture at 21.208 ms,
stopped the loading face at 25.051 ms and 2.120 mm when the front reached
y=440 mm, and completed full-fault rupture at 25.188 ms. It also exhibited a
1.997 ms stall near y=220 mm: this is an edge-loading-supported rupture, not a
fully spontaneous one. Early-stop and 7 ms hold pilots arrested near
y=137-147 mm, so the limitation must not be hidden by the case description.

The RSF parameters remain derived from the 5 mm cohesive-zone anchor in
`CohesiveZoneModel/Lc_estimate.py`, including shared
`D_c=0.000376505 mm`. The loading zone has `a=b=0.004` and is velocity
neutral. The conservative uncompressed estimate is 1.253 TB; the calibrated
LZF estimate is 1.196 TB and must remain below the configured 1.20 TB limit.
