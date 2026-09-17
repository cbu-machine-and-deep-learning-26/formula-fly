# Camera-to-hex resampling

The flyvis optic-lobe model receives a grayscale movie shaped
`(batch, time, 1, 721)`. `fly_driver.eyes.hex_resampler.HexResampler` converts
the project's declared `96 × 96 × 3` uint8 RGB camera frames to that format
without requiring flyvis at runtime.

## Geometry

The 721 columns form an axial hex grid of extent 15. Columns use the exact
`flyvis.utils.hex_utils.get_hex_coords(15)` ordering: `u` changes slowest, then
`v`. For each `(u, v)`, the corresponding Cartesian image offsets are:

```text
row offset    = 13 × (u + v / 2)
column offset = 13 × v
```

Image row zero is the top. Therefore negative row offsets are above center and
negative column offsets are left of center. At fixed `u`, increasing `v` moves
down and right; this fixes the lattice chirality as well as its orientation.
The implementation intentionally follows flyvis `BoxEye`, including its
float-to-integer truncation for half-row offsets.

The declared 96-pixel camera frame is bilinearly projected to flyvis's minimum
391-pixel sampling field, box-filtered with a 13-pixel kernel, and sampled at
the receptor centers. An input whose shape differs from the declared camera
shape is rejected; it is never silently resized into compliance.

RGB is converted with BT.601 luminance weights
`0.299 R + 0.587 G + 0.114 B`, scaled to float32 `[0, 1]`.

## Verification

Base tests prove shape, dtype, range, ordering, orientation, chirality, and
frame rejection without importing flyvis. When flyvis is installed, optional
tests also compare every output value with `BoxEye` and run moving gratings
through the pretrained network to gate T4/T5 direction selectivity.

Install and download the optional stack using the instructions introduced by
PR #40, then run:

```bash
export FLYVIS_ROOT_DIR="$HOME/.cache/flyvis"
pytest -q tests/eyes
```
