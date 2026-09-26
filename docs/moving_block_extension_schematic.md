# Moving-block extension toward -y: geometry for approval

This is a geometry proposal only. No simulation input, solver, or Slurm job
has been changed. The drawing is reproducible with
`draw_moving_block_extension_schematic.py` in this directory.

## Geometry

- Original moving block: x = 0..200 mm, y = 0..500 mm.
- Proposed moving block: x = 0..200 mm, y = -L_ext..500 mm.
- Stationary block unchanged: x = 200..345 mm, y = 0..550 mm.
- Original fault unchanged: x = 200 mm, y = 0..500 mm.
- Added material is continuously connected to the old moving block at y = 0;
  the dashed line in the drawing is not an additional contact interface.
- The extension is drawn with an illustrative length only. No sweep range,
  run IDs, or nominal extension length have been selected.

## Loading and contact

- Retain normal traction on x = 0 only for the original y = 0..500 mm face.
  The annotated 16 MPa is the existing reference load, not a new setting.
- The extension's side surfaces, y < 0, receive no normal traction and are free.
  This does NOT impose zero internal stress in the added material.
- Below y = 0 there is no stationary block, so there is no contact pair to
  which an RSF law should be applied. Mu = 0 in the drawing denotes the user's
  frictionless extension, not an artificial zero-friction overlap interface.
- Keep the existing RSF profile on the original 500 mm fault. Its profile
  must remain referenced to y = 0, not to the new moving-block origin.
- Move the prescribed shear-displacement face to y = -L_ext, over x = 0..200 mm,
  keeping the original +y direction. No displacement is prescribed on the
  now-internal y = 0 section. This is the assumed loading arrangement to confirm.
- Preserve the stationary outer-face normal-component supports: u_x = 0
  at x = 345 mm and u_y = 0 at y = 550 mm.

## Interpretation

The extension is assumed to retain the moving-block material unless changed
after approval. It represents additional series compliance, not an explicit
hydraulic-fluid model. Increasing length increases compliance, but does not
automatically increase stored energy when the imposed total displacement is
held fixed. In a simple axial-bar analogy, k = EA/L, U = F^2/(2k) at fixed
force, and U = k*u^2/2 at fixed displacement. The finite-block fault problem
and its stopping rule must be evaluated separately.

Implementation will need to mask the normal-load boundary by the original
fault coordinates, preserve the profile coordinates, and relocate the loading
face. Merely changing the block origin/dimensions would incorrectly extend
the currently whole-face normal load. None of these implementation changes
has been made yet.
