# TS0336-TS0351: continuous RSF nucleation and propagation

This 16-run campaign removes the locked-interface traction construction and
state inversion used in TS0320-TS0335. It starts from the TS0319 displacement-
controlled loading geometry and evolves the validated regularized ageing law
through the entire normal and shear history. No new friction law is introduced.

The campaign addresses two separate uncertainties: the effect of imposed
loading rate and the effect of the direct velocity response on nucleation and
propagation. The former slow front and the recent near-P-wave front do not by
themselves prove that either method produces the desired autonomous rupture.

## Matrix (exactly 16 independent GPUs)

| Loading/middle a | 75 ms ramp | 50 ms ramp | 30 ms ramp | 19.242255 ms ramp |
| --- | --- | --- | --- | --- |
| 0.0025 | TS0336 | TS0337 | TS0338 | TS0339 |
| 0.0050 | TS0340 | TS0341 | TS0342 | TS0343 |
| 0.0100 | TS0344 | TS0345 | TS0346 | TS0347 |
| 0.0200 | TS0348 | TS0349 | TS0350 | TS0351 |

All nominal ramps go from 0 to 2.45 mm with a half cosine, giving peak imposed
speeds of 51.31, 76.97, 128.28 and 200 mm/s. TS0343 retains the TS0319 material
and friction parameters and nominal ramp, but stops on the loading-end event
instead of waiting for full-fault slip. It is the direct control for that change.

For the loading and middle zones:

```
b - a = 0.024123527228205267
b * Dc = b_TS0319 * Dc_TS0319
V0 = V_init = 1e-4 mm/s
V_dynamic_calibration = 200 mm/s
f0 = 0.8
```

The steady friction curve over the calibrated range is preserved (0.8 at
V_init, 0.45 at 200 mm/s). Holding b*Dc fixes the original velocity-step
breakdown-work calibration because its velocity ratio is unchanged. This
does not fix the actual event's peak strength, nucleation size, breakdown work,
or speed. In particular, a changes the direct friction jump, while b and Dc
are co-calibrated; this is not a single-parameter experiment at fixed Dc.
The leading 30 mm VS law, including its Dc, is unchanged, and there is no chamfer.

## Loading and stop

- Normal: 16 MPa traction; 20 ms ramp plus 20 ms hold, explicit and undamped.
- Shear: displacement on moving-block-right in the same direction as TS0319.
- RSF is active throughout; theta starts at Dc/V_init and is never reset.
- No prestress_shear_displacement, quasistatic preload, or tangential stick phase.
- Freeze the loading face on the first V >= 500 mm/s at any station in y=5-25 mm.
  The slip guard is 1e-12 mm and there is no spatial coverage delay.
- Shear phase lasts 95 ms, giving at least 20 ms after the longest nominal ramp.
  Normal-phase creep is allowed and measured. An early normal-phase instability
  is diagnostic evidence, not silently removed from the history.

An instantaneous prescribed-velocity stop can launch an unloading wave even
though subsequent prescribed-face work is zero. The analysis must therefore
examine whether the front persists after stopping, rather than equating zero
boundary work with proof of a naturally sustained rupture. The 500 mm/s monitor
measures local interface relative slip rate, not loading-face velocity or Vr.

## Numerics, storage and diagnostics

Q4 mesh h=0.5 mm, dt=10 ns, float32, existing single-GPU projection. There are
13.5 million steps per run. Save 2 normal and 4400 shear bulk frames; save 1000
normal and 100000 shear interface frames (about 0.95 us shear sampling).
No animation or animation frames are generated on GB200.

The launch gate requires all 16 uncompressed dump estimates together to fit
under 1.4 TB and checks available disk space with a 50 GB reserve. The job
requests gb200-r1, four nodes with four GPUs each, one independent run per GPU,
no MPI, and 16 hours. Each run checkpoints every ten minutes and exits with a
checkpoint at the 15-hour runner limit, leaving scheduler margin.

`scripts/analyze_rsf_nucleation_sweep.py` writes per-run JSON/CSV to `stats/`.
It records normal-phase slip, state continuity, first 100/500/1000 mm/s arrivals,
1Dc and 0.05Dc arrivals, post-stop boundary work, post-stop kinetic energy,
and the maximum sampled V*dt/Dc. First-arrival fits exclude stations already
dynamic during normal loading and restrict y to 200 mm up to the VS region.
The latter resolution statistic is a lower bound on the true stepwise maximum.
Velocity thresholds, stress changes and accumulated slip must agree before
calling a fitted front a physical rupture. No sub-Rayleigh outcome is imposed.

Large inverted state ages alone were not proof of an implementation error:
traction inversion is valid for benchmark initial conditions, and f0 is a
reference coefficient, not a maximum friction. The unsupported PMMA preparation
history and abrupt constitutive switch were the issue this campaign removes.

Generate/check and submit:

```bash
python scripts/generate_ts0336_nucleation_sweep_cases.py --check
PREFLIGHT_ONLY=1 bash slurm/PMMA-RSF-GB200-R1-NUCLEATION-SWEEP.slurm
sbatch slurm/PMMA-RSF-GB200-R1-NUCLEATION-SWEEP.slurm
```

On an x86 login node, run the optional preflight with the x86 environment:
`PREFLIGHT_ONLY=1 ENV_PREFIX=/home/gauss112/.conda/envs/tatva bash slurm/PMMA-RSF-GB200-R1-NUCLEATION-SWEEP.slurm`.
Submit normally without that override; the scheduled GB200 tasks use the ARM environment.
