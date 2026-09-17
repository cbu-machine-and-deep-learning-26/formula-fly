"""Hex lattice geometry. Biological T4/T5 DS is gated on optional flyvis."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.hex_resampler import (
    HEX_COLUMNS,
    HEX_RADIUS,
    HexResampler,
    axial_disk,
    axial_to_pointy_xy,
)
from fly_driver.interface import DEFAULT_FRAME_SHAPE


def test_hex_column_count_is_flyvis_721() -> None:
    coords = axial_disk(HEX_RADIUS)
    assert len(coords) == HEX_COLUMNS == 721
    # Cube distance: max(|q|, |r|, |q+r|) ≤ radius
    q, r = coords[:, 0], coords[:, 1]
    s = -q - r
    assert int(np.max(np.maximum(np.abs(q), np.maximum(np.abs(r), np.abs(s))))) == HEX_RADIUS


def test_center_column_is_origin() -> None:
    coords = axial_disk()
    assert (coords == np.array([0, 0])).all(axis=1).any()


def test_column_order_is_r_then_q() -> None:
    coords = axial_disk(2)
    # First row is minimum r; within a row, q increases.
    rs = coords[:, 1]
    assert np.all(rs[1:] >= rs[:-1])
    for r in np.unique(rs):
        qs = coords[rs == r, 0]
        assert np.all(qs[1:] > qs[:-1])


def test_pointy_top_chirality_plus_q_is_plus_x() -> None:
    qx, qy = axial_to_pointy_xy(np.array([1.0]), np.array([0.0]))
    rx, ry = axial_to_pointy_xy(np.array([0.0]), np.array([1.0]))
    assert qx[0] > 0 and abs(qy[0]) < 1e-9
    # +r is 60 degrees: both x and y positive; cross product q × r > 0 (CCW).
    cross = qx[0] * ry[0] - qy[0] * rx[0]
    assert cross > 0
    assert rx[0] > 0 and ry[0] > 0


def test_resampler_output_length_and_dtype() -> None:
    resampler = HexResampler()
    frame = np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    frame[:, :, 1] = 128
    columns = resampler.resample(frame)
    assert columns.shape == (721,)
    assert columns.dtype == np.float32
    assert np.isfinite(columns).all()
    np.testing.assert_allclose(columns, 128 / 255.0, atol=1e-5)


def test_resampler_rejects_silent_resize() -> None:
    resampler = HexResampler()
    with pytest.raises(ValueError, match="no silent resize"):
        resampler.resample(np.zeros((48, 48, 3), dtype=np.uint8))


@pytest.mark.optional_flyvis
def test_t4_t5_direction_selectivity_requires_flyvis() -> None:
    """CLAUDE.md §11 gate: moving-edge / grating → T4/T5 preferred directions."""
    pytest.importorskip("flyvis")
    pytest.skip("T4/T5 direction-selectivity against flyvis checkpoints is Phase 1.")
