# TS0320-TS0335: traction-consistent RSF prestress sweep

This campaign retains the TS0319 geometry, material, 16 MPa normal traction,
RSF profile, 0.5 mm mesh, 10 ns time step, output cadence, and dynamic loading
rate. It changes only the initial shear prestress.

## Initial-condition construction

The original cases initialized the ageing state as `theta = Dc / V_init` while
the interface acquired a different shear traction during loading. That pair is
not generally an equilibrium point of the regularized RSF law and can create a
spurious strength jump when the shear phase begins.

For TS0320-TS0335, the 0-20 ms normal ramp remains explicit. From 20-30 ms the
loading face is moved with a half-cosine ramp while the interface is held
tangentially elastic and dynamic transients are damped; 30-40 ms is a hold.
At 40 ms the solver measures local shear and normal tractions and analytically
inverts the SCEC regularized RSF equation for `theta(y)` at
`V_init = 1e-4 mm/s`. Velocities are reset once, damping is disabled, and the
ordinary explicit RSF update begins. This preparation constructs an initial
condition; it is not counted as simulated frictional slip.

The prescribed prestress parameter is

```text
Pi = (tau0 / sigma_n - mu_residual) / (mu_peak - mu_residual),
mu_peak = f0 + a ln(V_dynamic / V_init).
```

The 16 values are 0.450, 0.475, ..., 0.825. Loading-face prestress
displacements, 1.57958-2.04113 mm, come from a linear `h -> 0` extrapolation
of h = 100, 50, 20, and 10 mm equilibrium pilots. Every dump records the
realized traction and state so the measured Pi, rather than this calibration,
can be used in interpretation. The diagnostics also record the fraction of
state values that hit the `float32` limits and the traction reconstruction
error, so a numerically invalid handoff cannot pass unnoticed.

## Dynamic event and stop

Every case then adds the same 2.45 mm half-cosine displacement increment over
19.242255 ms, with a 200 mm/s peak prescribed speed. The loading face freezes
on the same integration step where any station in y = 5-25 mm first has both
new dynamic slip and `|V| >= 500 mm/s`. There is no spatial coverage condition.
Propagation after this stop is therefore powered by stored elastic energy.

Each case retains 4400 bulk and 50000 interface shear frames. The Slurm job
runs exactly 16 independent one-GPU tasks and rejects a combined estimate over
1.4 TB before launch. Submit with:

```bash
sbatch slurm/PMMA-RSF-GB200-R1-PRESTRESS-SWEEP.slurm
```
