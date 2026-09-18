# Vendored track data

## `silverstone_centerline.csv`

The centerline geometry the MuJoCo practice track is built from (GH-16).

| | |
|---|---|
| **Source** | [TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database), `tracks/Silverstone.csv` |
| **Upstream licence** | **LGPL-3.0** — see `LICENSE-LGPL-3.0.txt` in this directory |
| **Copyright** | Technische Universität München, Institute of Automotive Technology |
| **Retrieved** | 2026-09-17, unmodified |
| **Columns** | `x_m, y_m, w_tr_right_m, w_tr_left_m` — centerline position and track half-widths to each side, all in metres |
| **Rows** | 1178 points at ~5 m spacing |

### Licence note

The rest of this repository is MIT (see the root `LICENSE`). **This one data file is not.**
It is redistributed under LGPL-3.0, which is why the licence text sits next to it.

We only *read* this file at runtime — we do not link against TUMFTM code, and nothing in
`fly_driver/` is a derivative work of it — so the copyleft does not propagate into our source.
Keeping the file in its own directory with its own licence and this notice is what keeps that
boundary obvious to anyone auditing the repo later. LGPL-3.0 incorporates GPL-3.0 by reference;
the full GPL text is at <https://www.gnu.org/licenses/gpl-3.0.txt>.

If we ever need the repo to be uniformly MIT, the fix is to stop vendoring and download this
file at setup time instead — at the cost of a network dependency in CI and non-deterministic
geometry if upstream changes.

### Why vendored rather than downloaded

Deterministic geometry. Every teammate, every CI run, and every Spark node loads byte-identical
track data, so a lap time measured on one machine means the same thing on another. A download
step would also put a network failure between a fresh clone and a passing test suite.

### Verification

Checked on import against the real circuit (see `tests/test_centerline.py`):

- Closed-loop length **5886.8 m** against Silverstone GP's official **5891 m** — 0.07% off,
  which is what you expect from ~5 m polyline sampling of a curved circuit
- Point spacing 4.68–5.16 m, mean 5.00 m
- Track half-widths 5.4–9.0 m per side
