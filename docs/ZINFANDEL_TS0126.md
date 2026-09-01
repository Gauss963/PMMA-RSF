# TS0126 on zinfandel

This note records the reproducible CPU/MPI setup for the sparse-output TS0126
completion test. The simulation physics match `rsf_0126_q4_chamfer20x5_12h.toml`;
only output sampling is reduced to respect the 100 GB home quota.

## Scheduler and storage

- Follow the site [Slurm guidance](https://hpc.aiengineer.tw/scheduling.html).
- Use `cpu-2g`; the 96-core node has approximately 512 GB RAM and permits a
  28-day wall time.
- The production script reserves 450 GB and one 96-core node.
- `cases/rsf_0126_q4_chamfer20x5_zinfandel_sparse.toml` conservatively estimates
  approximately 1.0 GB of output. Check `df -h "$HOME"` before submission; the
  project runner also preserves a 50 GB free-space reserve.
- Only MPI rank 0 writes persistent HDF5/checkpoint files. Other ranks write to
  node-local `/tmp` and remove those copies after a clean completion or checkpoint.

## Environment

Create the Python 3.12 environment without using the Anaconda default channels:

```bash
conda create -y -n tatva -c conda-forge --override-channels python=3.12
conda activate tatva
```

Build `mpi4jax` against the same Intel MPI module used at runtime. Do not load
the site's `opt` module here: its include variables duplicate `/usr/include`
and break the C++ `#include_next` search used while building `mpi4jax`.

```bash
module purge
module load mpi/2021.15
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
MPICC="$(command -v mpicc)" python -m pip install -e '.[mpi,dev]'
```

The web-page example currently names `mpi/latest`, but that alias is not
available on zinfandel. `mpi/2021.15` was verified with both `mpi4py` and
`mpi4jax`.

## Validation

Run the MPI element-assembly test with at least two ranks:

```bash
module purge
module load mpi/2021.15
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
mpirun -n 2 -ppn 2 python -m pytest -q tests/test_pmma_mpi.py
```

`cases/rsf_mpi_smoke.toml` exercises the complete PMMA contact/RSF/HDF5 path.
Compare its serial and MPI outputs with:

```bash
python scripts/compare_mpi_smoke.py \
  runs-zinfandel/smoke-serial/data/simulation.h5 \
  runs-zinfandel/smoke-mpi2/data/simulation.h5 \
  --json runs-zinfandel/mpi-smoke-comparison.json
```

The 2026-09-01 validation compared 47 numeric datasets and passed every
field-specific tolerance. The largest bulk-stress difference was
`2.41e-6 MPa`; the largest friction-strength difference was `6.78e-11`.

The production mesh/kernel scaling benchmark used the same 96-core node and
4,001 explicit steps for every layout:

| Layout | Wall time | Speedup vs serial |
| --- | ---: | ---: |
| Serial | 922.6 s | 1.00x |
| 4 MPI ranks x 24 threads | 550.9 s | 1.67x |
| 8 MPI ranks x 12 threads | 536.2 s | 1.72x |
| 12 MPI ranks x 8 threads | 464.0 s | 1.99x |
| 16 MPI ranks x 6 threads | 356.8 s | 2.59x |
| 24 MPI ranks x 4 threads | 413.3 s | 2.23x |

Sixteen ranks are the measured optimum. Twenty-four ranks add collective
communication without increasing useful CPU throughput. The full sparse run
is expected to take approximately 10-12 days; the 28-day allocation leaves
enough margin for output and checkpoint overhead.

The serial and 16-rank outputs from this production-size benchmark also passed
all 47 dataset comparisons. The maximum bulk-stress difference was
`2.91e-5 MPa`; the global relative differences were `2.67e-4` for slip rate
and `6.21e-6` for RSF state. The comparison report is written to
`runs-zinfandel/bench-serial-vs-mpi16.json` on zinfandel.

## Submission

Check out the validation branch and submit the production script:

```bash
git switch zinfandel-ts0126
git pull --ff-only
sbatch slurm/PMMA-RSF-ZINFANDEL-CPU.slurm
```

The script defaults to:

- case: `cases/rsf_0126_q4_chamfer20x5_zinfandel_sparse.toml`
- run directory: `runs-zinfandel/TS0126`
- resources: 96 cores, 450 GB, 28 days
- MPI layout: 16 ranks x 6 threads
- output estimate: approximately 1.0 GB
- checkpoint interval: 10 minutes, evaluated at short output-chunk boundaries

Override the MPI layout without changing the script:

```bash
sbatch --export=ALL,MPI_RANKS=12 slurm/PMMA-RSF-ZINFANDEL-CPU.slurm
```

`MPI_RANKS` must divide 96. Each rank uses `96 / MPI_RANKS` CPU threads, and
the script pins each Intel MPI rank to its OpenMP domain.

Monitor the run and quota with:

```bash
squeue -u "$USER"
tail -f runs-zinfandel/TS0126/logs/job.log
df -h "$HOME"
```

To resume a clean wall-time checkpoint, submit the same case and run directory:

```bash
sbatch --export=ALL,RESUME_DIR="$HOME/PMMA-RSF/runs-zinfandel/TS0126" \
  slurm/PMMA-RSF-ZINFANDEL-CPU.slurm
```
