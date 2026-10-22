"""ControlVector onto AssettoCorsaGym's absolute action space, without AC installed."""

from __future__ import annotations

import mmap
from types import SimpleNamespace

import numpy as np

from fly_driver.bridge.assetto import CaptureReader, to_ac_action
from fly_driver.interface import ControlVector

# dallara_f317's norm_steer_at_max, from upstream's steer_map.csv.
F317_MAX_STEER = 0.5341796875


def test_extremes_map_to_full_lock_and_released_or_full_pedals() -> None:
    np.testing.assert_allclose(
        to_ac_action(ControlVector(-1.0, 0.0, 1.0), F317_MAX_STEER), [-F317_MAX_STEER, -1, 1]
    )
    np.testing.assert_allclose(
        to_ac_action(ControlVector(1.0, 1.0, 0.0), F317_MAX_STEER), [F317_MAX_STEER, 1, -1]
    )


def test_neutral_is_centred_with_both_pedals_released() -> None:
    np.testing.assert_allclose(to_ac_action(ControlVector.neutral(), F317_MAX_STEER), [0, -1, -1])


def _camera(images: list[np.ndarray], active: int) -> SimpleNamespace:
    """The plugin's dual buffer layout: 4-byte active index, then two images."""
    height, width, channels = images[0].shape
    size = height * width * channels
    shm = mmap.mmap(-1, 4 + 2 * size)
    shm.write(active.to_bytes(4, "little"))
    for image in images:
        shm.write(image.tobytes())
    return SimpleNamespace(
        dual_shm=shm,
        image_size=size,
        shm_image_height=height,
        shm_image_width=width,
        channels=channels,
    )


def test_a_black_frame_is_read_whole() -> None:
    """Upstream strips trailing zero bytes, so an all-black frame came back empty and crashed."""
    black = np.zeros((96, 96, 3), np.uint8)
    lit = np.full((96, 96, 3), 200, np.uint8)
    reader = CaptureReader(_camera([lit, black], active=1))
    _, _, image = reader()
    assert image.shape == (96, 96, 3) and not image.any()
    assert reader.misses == 0


def test_the_active_index_picks_the_buffer() -> None:
    first = np.full((4, 4, 3), 10, np.uint8)
    second = np.full((4, 4, 3), 20, np.uint8)
    _, _, image = CaptureReader(_camera([first, second], active=0))()
    np.testing.assert_array_equal(image, first)


def test_a_bad_read_reuses_the_last_frame_and_is_counted() -> None:
    lit = np.full((4, 4, 3), 200, np.uint8)
    camera = _camera([lit, lit], active=0)
    reader = CaptureReader(camera)
    _, _, good = reader()
    camera.dual_shm.seek(0)
    camera.dual_shm.write((7).to_bytes(4, "little"))  # a torn index: no such buffer
    _, _, image = reader()
    np.testing.assert_array_equal(image, good)
    assert reader.misses == 1
