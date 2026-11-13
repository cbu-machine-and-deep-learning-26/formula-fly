"""One flyvis network for many tracks gives each track exactly its own eye's features (GH-27)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flyvis")

from fly_driver.eyes.batched import BatchedFlyvisEye, SharedEye  # noqa: E402
from fly_driver.eyes.flyvis_eye import FlyvisEye, resolve_checkpoint_dir  # noqa: E402
from fly_driver.eyes.pixel_eye import PixelEye  # noqa: E402


@pytest.fixture(scope="module")
def eyes() -> tuple[list[FlyvisEye], BatchedFlyvisEye]:
    try:
        resolve_checkpoint_dir()
    except FileNotFoundError:
        pytest.skip("pretrained flyvis checkpoint not downloaded")
    references = [FlyvisEye(device="cpu") for _ in range(3)]
    for eye in references:
        eye.reset()
    return references, BatchedFlyvisEye(FlyvisEye(device="cpu"), 3)


def test_batched_tracks_match_their_own_eyes_through_resets_and_partial_steps(eyes) -> None:
    references, batched = eyes
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (8, 3, 96, 96, 3), dtype=np.uint8)
    for step, batch in enumerate(frames):
        if step == 3:
            references[1].reset()
            batched.reset([1])
        if step == 5:
            # Only tracks 0 and 2 see this frame, as when a capped episode bootstraps.
            rows = [0, 2]
            got = batched.encode(batch[rows], rows)
            want = np.stack([references[i].encode(batch[i]) for i in rows])
        else:
            got = batched.encode(batch)
            want = np.stack(
                [eye.encode(frame) for eye, frame in zip(references, batch, strict=True)]
            )
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_a_reset_is_the_grey_warm_up(eyes) -> None:
    _, batched = eyes
    fresh = FlyvisEye(device="cpu")
    fresh.reset()
    frame = np.full((1, 96, 96, 3), 128, dtype=np.uint8)
    batched.reset([0])
    np.testing.assert_allclose(batched.encode(frame, [0])[0], fresh.encode(frame[0]), atol=1e-6)


def test_wrong_frame_shape_is_refused(eyes) -> None:
    _, batched = eyes
    with pytest.raises(ValueError, match="no implicit resize"):
        batched.encode(np.zeros((3, 64, 64, 3), dtype=np.uint8))


def test_a_shared_stateless_eye_encodes_every_frame() -> None:
    shared = SharedEye(PixelEye(frame_shape=(96, 96, 3), downsample=8))
    frames = np.random.default_rng(0).integers(0, 256, (4, 96, 96, 3), dtype=np.uint8)
    assert shared.encode(frames).shape == (4, shared.feature_dim)
