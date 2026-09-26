# Unloaded moving-block extension: TS0352--TS0367

## Hypothesis and limits

Test whether additional system compliance can sustain a forward rupture after
the displacement actuator has stopped. This is **not** an assumption that
continued loading is the only cause of slow rupture. RSF nucleation, finite
fault length, prestress distribution and velocity-strengthening regions can
also arrest or slow a front. See McLaskey and Yamashita (2017),
<https://doi.org/10.1002/2016JB013681>, for laboratory examples of loading-dependent
slow/fast slip and finite nucleation/available-energy limitations.

The first sweep changes **only extension length**, not elastic modulus,
friction, the nominal displacement program or the stop criterion:

- TS0352--TS0367 correspond to `L_ext = 0, 2, ..., 30 mm` (exactly 16 cases).
- Baseline input: `cases/rsf_0343_nucleation_08.toml`.
- Moving block: `x=0..200`, `y=-L_ext..500 mm`, continuous PMMA throughout.
- Stationary block: unchanged, `x=200..345`, `y=0..550 mm`.
- Fault: unchanged, `x=200`, `y=0..500 mm`; original RSF profiles unchanged.
- No external normal traction and no contact pairs on `y<0`.
- Normal load: 16 MPa on `x=0, y=0..500`, not on the extension.
- Prescribed +y displacement: the entire new bottom face `y=-L_ext`.
- At positive extension, `y=0` is internal, without a displacement constraint.
- Both the normal traction and contact quadrature weights at `y=0` retain the
  original half-segment contribution. Masking full-face nodal weights would
  incorrectly include half of a negative-y segment.

`geometry.moving.origin` and `dimensions` describe the original body. The new
`loading_extension_length` field grows its mesh without shifting fault material
profiles, physical stop coordinates, or the normal-load interval. Extension
length must align with mesh cells.

## Order-of-magnitude compliance estimate

For an axial bar analogy, with unit out-of-plane thickness, E=7662 MPa and
A=200 mm^2:

    k_ext = E A / L_ext
    delta_ext = F L_ext / (E A)
    U_ext = F^2 L_ext / (2 E A)

At 30 mm, k_ext is approximately 51,080 N/mm. The first saved stopped frame in
TS0343 has a reaction of 1723.738 N per mm thickness (previous frame 1843.581).
Using 1723.738 gives delta_ext about 0.03375 mm and U_ext about 29.08 N mm per mm
thickness. This is about 18% of TS0343's roughly 160.27 incremental elastic-plus-
interface energy between normal end and actuator arrest, **not** 18% of its
total stored normal-load energy.

These are illustrative estimates at a fixed force. The actual model is a
2-D constrained body, not a slender uniform-stress bar; the force and arrest
time change with L_ext. Added length does not guarantee increased usable energy
under a fixed-displacement protocol. Measure extension energy in the dump.
The 0--30 mm range is a moderate first test, not a claim that it reproduces a
particular hydraulic machine stiffness. If the effect is too small, a later
sweep should be guided by measured loading-system compliance.

## Unchanged loading and numerics

- Full undamped explicit integration, no quasi-static stage, no state reset,
  no traction-consistent handoff, no artificial normal-phase shear prestress.
- Normal phase 40 ms; normal ramp 20 ms.
- Nominal shear ramp: half-cosine, 0 to 2.45 mm in 19.242255 ms (peak 200 mm/s).
- Freeze displacement immediately when a monitored fault station in original
  `y=5..25 mm` first reaches 500 mm/s and accumulated slip exceeds 1e-12 mm.
- Shear phase 95 ms, including time after actuator arrest.
- Q4, h=0.5 mm, dt=10 ns, float32, operator batch size 65536.
- Loading/middle a=0.005, b=0.029123527228205267, Dc=0.00042852936846183833 mm.
- Leading 30 mm remains weakly VS; no change to transition or chamfer settings.
- No animations or animation frames on the GPU allocation.

## Common numerical correction and control

All new cases use **log-speed bisection** for the regularized RSF projection
`V + A*tau(V,theta) = |V_free|`. The old 64 linear bisections could not reach
the exponentially small near-sticking root. In a manufactured PMMA case with
true tau/sigma=0.1 and a=0.005, the old returned strength was about 0.644 sigma.
The velocity was negligible, but the reported traction/coefficient was wrong.

The friction law itself is unchanged. The new solver brackets and solves in
log speed and evaluates traction there, even if the physical speed underflows
float32. Regression tests check momentum residuals, known roots over many
decades, open contact and SCEC-scale material parameters. These tests are not
a new full TPV101/102 benchmark.

Use **TS0352**, not old TS0343 alone, as the zero-extension control: it shares
the numerical correction, energy diagnostics, and all other settings with the
15 nonzero extensions. Historical dumps are not rewritten.

## Diagnostics and evaluation

The 17-column high-rate history adds:

- `extension_elastic_energy`: integration of strain energy over original `y<0`
  elements, computed from the same quadrature as total strain energy.
- `normal_loading_coordinate`: force-weighted normal displacement of the
  loaded face, conjugate to the total normal traction force.
- `normal_external_force`: applied traction force, allowing normal work to be
  separated from remote shear work. This column is zero/not a reaction
  diagnostic for normal-displacement mode; this sweep uses stress mode only.

After each completed run, `scripts/analyze_rsf_extension_sweep.py` saves JSON
metrics and station-arrival CSVs under `stats`. It reports post-arrest shear
work, normal work, extension-energy release, total energy change, peak kinetic
energy and interface first crossings at 100, 500 and 1000 mm/s.

Success requires a spatially coherent forward front **after** arrest, near-zero
post-arrest remote shear work, release of stored energy, and slip over the
whole fault. A crossing of one Dc alone or a poorly correlated arrival-time
fit does not establish complete dynamic rupture. Inspect fit coverage and R^2,
compare thresholds and exclude the VS region from a middle-front speed fit.
Instantaneous actuator arrest remains identical to the baseline; any resulting
wave must not be mislabeled as a naturally propagating front.

## Capacity and submission

Bulk frames: 2 normal + 4400 shear per run. Interface: 1000 normal + 100000
shear, retaining 0.95 microsecond nominal shear sampling. Estimated raw dump
size is about 1.283 TB for the sweep (78.91--81.45 GB per case), below 1.4 TB.
Degrees of freedom grow from 1,443,584 to 1,491,704 (about 3.3%). The mesh cost
increase is small; timing changes from the corrected solver still need to be
measured on GB200.

Launcher: `slurm/PMMA-RSF-GB200-R1-EXTENSION-SWEEP.slurm`.
GB200-r1, 4 nodes, 16 independent tasks, one GPU/run, no MPI, 16-hour wall limit.
Checkpoint every 10 minutes; request a clean stop at 15 hours with a buffer
before Slurm termination. Storage gate reserves 50 GB beyond estimated dumps.
Each GPU rank first runs `scripts/validate_rsf_extension_backend.py`: a small
temporary-mesh test of near-rest traction, loaded/contact extents, energy output,
and actuator arrest. Temporary data are removed automatically. A failed check
prevents that rank from starting the production dump. Local regression tests
passed (100 tests); the same standalone smoke test passed on the local CPU.

    python scripts/generate_ts0352_extension_sweep_cases.py --check
    PREFLIGHT_ONLY=1 ENV_PREFIX=/home/gauss112/.conda/envs/tatva \
      bash slurm/PMMA-RSF-GB200-R1-EXTENSION-SWEEP.slurm
    sbatch slurm/PMMA-RSF-GB200-R1-EXTENSION-SWEEP.slurm

Run output: `/work/gauss112/tatva/runs/TS0352` through `TS0367`.
