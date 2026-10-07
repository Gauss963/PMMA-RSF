# LSW 0114 with a 16 MPa normal stress boundary

Status as of 2026-10-07 (Asia/Taipei): **production not submitted**. F1 rejects
submission with an insufficient-credit error, but `wallet MST113389` reports a
positive balance of **941.7681 SU**. The conflicting balances require investigation
by F1 support; actual insufficient funds have not been established. Do not attempt
an alternative submission method or unrelated account to bypass the rejection.

## Configuration provenance

Input: `cases/lsw_0114_normal_stress_16mpa.toml`.

The archived 0114 run no longer contains its summary or dump. Reconstruction uses
the archived 0116 `stats/summary.json` and the old displacement notebook's explicit
0114 restoration notes. These agree on the effective LSW settings. In particular,
the old 0114 directory label `hold1ms` is stale: its actual shear window was 30 ms
(10 ms ramp plus 20 ms observation), with 4,800 normal and 95,000 shear frames.
The archived 0116 summary also confirms that its normal loading was displacement
controlled despite its `normal16mpa` directory label.

Preserved parameters:

- Moving block: 200 x 500 mm; stationary block: 145 x 550 mm.
- 1 mm Tri3 mesh: 181,147 nodes, 359,500 elements, 362,294 displacement DOFs.
- E = 7662 MPa, nu = 0.2, density = 1.148e-9 in the original mm/MPa/s units.
- LSW: mu_s = 0.8, mu_k = 0.45, middle D_c = 0.00629795 mm.
- Loading-end nucleation length 120 mm, D_c at the loading end 0.315 mm.
- Original terminal creep: length 10 mm, transition 10 mm, mu_s/mu_k = 0.8/0.45,
  relaxation time 0.05 ms. No RSF or Prakash-Clifton law is introduced.
- Fully explicit, undamped normal phase 40 ms with a 20 ms linear ramp.
- Moving-block-right shear displacement 0 to 1.76 mm, 10 ms half-cosine ramp.
- Original loading stop: first local D_c crossing within y = 0 to 440 mm;
  no new velocity threshold, prestress, or quasi-static handoff.
- dt = 1.2492928633545953e-8 s, normal/tangential penalties 76620/7662.
- Original full bulk fields and 99,800 saved frames; uncompressed bulk estimate
  1.437 TB before LZF compression. Only MPI rank zero writes data/checkpoints.

The intended physical change is solely the normal BC: uniform 16 MPa traction on
moving-block-back replaces prescribed normal displacement. The archived normal
displacement 0.691620986687549 mm is retained only as inactive metadata. A 16 MPa
boundary traction does not imply equality with the old displacement-controlled
interface normal stress (the archived 0116 final mean was about 19.54 MPa).

## Implementation and checks

The existing PMMA solver is reused. The tested CPU MPI approach partitions elastic
element integration and sums energy and nodal gradients, with replicated contact,
constraints, and time integration. It does not reduce mesh or temporal resolution.
The current MPI entry point is deliberately limited to undamped explicit LSW.
Existing single-process RSF execution remains supported and unchanged by default.

- Local regression: all 50 tests in `test_pmma_cases.py` and both new LSW tests pass.
- F1 allocation 319: 4-rank versus serial small-mesh comparison passed.
- Maximum small-mesh moving stress difference: 7.63e-6 MPa.
- Maximum small-mesh moving displacement difference: 3.73e-9 mm.
- F1 MPI checkpoint/resume comparison: all ten checked datasets match exactly.
- Full-mesh serial benchmark completed, 4,000 steps per phase.
- The following 7-rank x 8-thread timing launch was rejected by MPI because its
  binding crossed a CPU package boundary. It never entered the solver.
- Fixed benchmark layouts: 8 x 7, 14 x 4, and 28 x 2, with package-aware binding.
  These full-mesh MPI timings and full-mesh serial/MPI comparison remain pending.

F1 benchmark results:
`/work1/gauss112/PMMA-LSW-MPI/runs/benchmarks/lsw-0114-319/`.

## Runtime estimate

The serial benchmark used one rank with eight CPU threads, including full-field
compression/write time, excluding first-time JIT compilation in each chunk shape:

| Phase | Timed steps | Wall seconds | Extrapolated production hours |
| --- | ---: | ---: | ---: |
| Normal | 2400 | 11.1942 | 4.15 |
| Shear | 3948 | 30.3747 | 5.13 |
| Total | | | **9.28** |

Production requires 3,201,812 normal and 2,401,359 shear steps. This is a short-test
estimate, not a measured complete run; actual compression and shared filesystem
load can change the cost. The benchmark accelerates its loading schedules solely
for software/performance testing and is not a physical rupture experiment.
Do not claim an MPI speedup until the remaining measurements are available.

## F1 environment and next steps

Source checkout: `/work1/gauss112/PMMA-LSW-MPI`, branch
`codex/lsw0114-stress-mpi`. The existing `/work1/gauss112/tatva` checkout and its
postprocessing environment were not changed.

Environment: `/home/gauss112/.venvs/tatva-lsw-mpi`, based on the existing Tatva
Python 3.12 environment with isolated JAX 0.9.2, mpi4py 4.1.1, and mpi4jax 0.9.0;
modules `gcc/11.2.0` and `openmpi/5.0.2`. mpi4jax warns that its officially tested
JAX maximum is 0.9.1; the actual JAX 0.9.2 small-mesh and resume checks passed.

Submission initially encountered a scheduler script/environment I/O error. A
Slurm interactive allocation was used for the successful partial benchmark.
The subsequent submission explicitly reports insufficient iService credits.
Rechecking on 2026-10-07 after the user supplied a positive wallet balance confirms
a discrepancy, not a demonstrated need to top up:

- Direct `wallet MST113389`: **941.7681 SU**, confirmed on the active F1 login node.
- `sbatch --account=MST113389 --time=00:30:00 ...`: rejected with
  `You do not have enough credits in your iService wallet -50052.955500000004`.
- Retrying with the canonical Slurm account spelling `mst113389` returns the same
  error. `sacctmgr` confirms this account is associated with `gauss112` on `f1`.
- No `SLURM*` or `SBATCH*` environment overrides were present. `sbatch` resolves to
  `/usr/bin/sbatch`; the controller reports `JobSubmitPlugins=lua`.
- No job ID was created; `squeue -u gauss112` was empty during these checks.

The reason for the disagreement is not yet known. In particular, stale data or an
account mapping problem are possibilities, not verified diagnoses. F1 support
needs to reconcile the wallet query with the submission-side credit check. Do not
assume that another top-up, a different launch route, or an unrelated account is
the appropriate remedy. SU-to-cost conversion has not been established here, so
the positive balance alone does not prove that the complete run is affordable.

After F1 resolves the submission-side credit discrepancy:

1. Finish the package-aware benchmark, reusing the completed results:

   ```bash
   sbatch --export=ALL,RESULTS=/work1/gauss112/PMMA-LSW-MPI/runs/benchmarks/lsw-0114-319 /work1/gauss112/PMMA-LSW-MPI/slurm/PMMA-LSW-F1-BENCHMARK.slurm
   ```

2. Require `full-mesh-validation.json` to pass; compare measured MPI runtimes and
   choose ranks/threads and a time limit with headroom. The production script's
   current 48-hour limit is a placeholder ceiling, not a runtime estimate.
3. Reserve a new run ID without overwriting any existing run. TS0384 was available
   at the last check, but it has not been reserved or started.
4. Submit `slurm/PMMA-LSW-F1.slurm` with `RUN_DIR` set to that directory. Verify the
   job ID and queue state rather than relying on `sbatch --test-only`.

Checkpoints are written every ten minutes and before the time limit (ten-minute
headroom). Resubmission to the same directory resumes automatically and validates
the original input. No animations or other costly postprocessing run automatically.
