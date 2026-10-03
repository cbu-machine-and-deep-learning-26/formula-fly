"""The fly's-eye view in the drive tool (GH-75).

Three layers, tested at the cheapest level each can be:

* :class:`EyePicture` and the placement maths are pure numpy -- no eye, no OpenGL.
* :class:`FlyEyeView` with a stand-in eye renders a real camera, so it needs an OpenGL
  context (``render``) but not flyvis: what it hands the eye, and when, is the contract.
* One test runs the real pretrained eye on the real track and checks the thing the view is
  for: a moving car shows more motion than a parked one. It needs flyvis, the checkpoint
  and a renderer, and skips without them.

The module needs torch (the hex lattice is torch code), so ``tests/conftest.py`` skips it
on the base install, like the other eye tests.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fly_driver.analysis.hex_plots import HexRaster
from fly_driver.analysis.motion_percept import T4_READOUTS, T5_READOUTS
from fly_driver.eye_view import (
    EYE_VIEW_KEY,
    PANEL_GAP_PX,
    PANEL_PX,
    VISIBLE_FRACTION,
    EyePicture,
    EyeViewMode,
    FacetLayout,
    FlyEyeView,
    big_view_side,
    place_overlay,
)
from fly_driver.eyes.hex_resampler import HEX_COLUMN_COUNT
from fly_driver.hud import HEIGHT as HUD_HEIGHT
from fly_driver.hud import WIDTH as HUD_WIDTH

READOUTS = T4_READOUTS + T5_READOUTS
FEATURES = len(READOUTS) * HEX_COLUMN_COUNT
#: A column near the middle of the lattice.
COLUMN = HEX_COLUMN_COUNT // 2


def _features(**active: float) -> np.ndarray:
    """Zero features except ``active`` (readout -> value) in :data:`COLUMN`."""
    features = np.zeros(FEATURES, dtype=np.float32)
    for name, value in active.items():
        features[READOUTS.index(name) * HEX_COLUMN_COUNT + COLUMN] = value
    return features


def _picture() -> EyePicture:
    return EyePicture(READOUTS, np.zeros(FEATURES, dtype=np.float32))


def _column_pixels(image: np.ndarray, column: int = COLUMN) -> np.ndarray:
    """Every pixel of ``image`` that paints ``column``."""
    return image[HexRaster(PANEL_PX).index == column]


class TestModes:
    def test_the_key_cycles_corner_big_fly_off_and_round_again(self):
        mode = EyeViewMode.CORNER
        seen = [mode := mode.next() for _ in range(4)]
        assert seen == [EyeViewMode.BIG, EyeViewMode.FLY, EyeViewMode.OFF, EyeViewMode.CORNER]

    def test_the_key_is_f10(self):
        """GLFW_KEY_F1 is 290. MuJoCo's viewer binds F1-F6, F8, F9 and acts on them as well
        as passing them on, so moving this onto one of those would fire a viewer command."""
        assert EYE_VIEW_KEY == 299


class TestEyePicture:
    def test_an_eye_at_rest_shows_no_motion(self):
        picture = _picture()
        picture.update(np.zeros(FEATURES), np.full(HEX_COLUMN_COUNT, 0.5))
        assert picture.motion_strength == 0.0
        assert not picture.motion_rgb.any()

    def test_rightward_motion_paints_its_column_red(self):
        picture = _picture()
        picture.update(_features(T4b=1.0), np.zeros(HEX_COLUMN_COUNT))
        pixels = _column_pixels(picture.motion_image())
        assert len(pixels), "the column paints no pixels; the raster lookup is wrong"
        assert (pixels[:, 0] > 0.9).all() and (pixels[:, 1:] < 0.1).all()
        assert picture.motion_strength > 0.0

    def test_leftward_motion_paints_its_column_cyan(self):
        picture = _picture()
        picture.update(_features(T5a=1.0), np.zeros(HEX_COLUMN_COUNT))
        pixels = _column_pixels(picture.motion_image())
        assert (pixels[:, 0] < 0.1).all() and (pixels[:, 1:] > 0.9).all()

    def test_motion_is_measured_from_the_resting_baseline(self):
        """The view's rest is the eye settled on the still scene, not zero activity."""
        picture = EyePicture(READOUTS, _features(T4b=1.0))
        picture.update(_features(T4b=1.0), np.zeros(HEX_COLUMN_COUNT))
        assert picture.motion_strength == 0.0

    def test_the_retina_is_drawn_as_grey(self):
        picture = _picture()
        picture.update(np.zeros(FEATURES), np.full(HEX_COLUMN_COUNT, 0.5))
        inside = picture.retina_image()[HexRaster(PANEL_PX).is_inside]
        np.testing.assert_allclose(inside, 0.5)

    def test_the_filter_dims_the_retina_under_the_motion(self):
        picture = _picture()
        picture.update(_features(T4b=1.0), np.ones(HEX_COLUMN_COUNT))
        image = picture.filter_image()
        still = image[HexRaster(PANEL_PX).index == 0][0]
        moving = _column_pixels(image)[0]
        assert np.allclose(still, still[0]) and 0.0 < still[0] < 1.0, "still retina not dimmed"
        assert moving[0] > moving[1], "the moving column is not tinted towards red"

    def test_corner_image_is_retina_and_motion_side_by_side(self):
        image = _picture().corner_image()
        assert image.shape == (PANEL_PX, 2 * PANEL_PX + PANEL_GAP_PX, 3)
        assert image.dtype == np.uint8

    @pytest.mark.parametrize("side", [1, 191, 192, 500, 673])
    def test_big_image_is_exactly_the_requested_square(self, side):
        """set_images raises unless the image is exactly the rect's size."""
        image = _picture().big_image(side)
        assert image.shape == (side, side, 3) and image.dtype == np.uint8

    def test_big_image_rejects_a_non_positive_side(self):
        with pytest.raises(ValueError):
            _picture().big_image(0)

    def test_a_retina_of_the_wrong_size_is_refused(self):
        with pytest.raises(ValueError):
            _picture().update(np.zeros(FEATURES), np.zeros(HEX_COLUMN_COUNT - 1))

    def test_clear_forgets_the_last_frame(self):
        picture = _picture()
        picture.update(_features(T4b=1.0), np.ones(HEX_COLUMN_COUNT))
        picture.clear()
        assert picture.motion_strength == 0.0
        assert not picture.motion_rgb.any() and not picture.retina.any()


class TestTheFlyView:
    """The whole window as the fly's retina: grey facets, dark lines between them.

    Laid out from a small lattice here (64 px); the view itself uses a finer one, and the
    layout maths does not care which.
    """

    RASTER = HexRaster(64)

    def _layout(self, width: int = 400, height: int = 300) -> FacetLayout:
        return FacetLayout(width, height, self.RASTER)

    def test_the_image_is_exactly_the_window(self):
        """set_images raises unless the image is exactly the rect's size."""
        image = self._layout(400, 300).paint(np.full(HEX_COLUMN_COUNT, 0.5))
        assert image.shape == (300, 400, 3) and image.dtype == np.uint8

    def test_the_eye_is_centred_in_the_part_assumed_visible(self):
        layout = self._layout(1000, 500)
        visible_width, visible_height = 1000 * VISIBLE_FRACTION, 500 * VISIBLE_FRACTION
        assert layout.side == int(visible_height)
        assert layout.left + layout.side / 2 == pytest.approx(visible_width / 2, abs=1)
        assert layout.top == 0

    def test_each_facet_is_one_flat_grey_level(self):
        retina = np.linspace(0.0, 1.0, HEX_COLUMN_COUNT)
        layout = self._layout()
        image = layout.paint(retina)
        square = image[
            layout.top : layout.top + layout.side, layout.left : layout.left + layout.side
        ]
        facet = square[layout.square_index == COLUMN]
        assert len(facet), "the column paints no pixels"
        assert (facet == facet[0]).all(), "a facet is not one flat colour"
        assert facet[0][0] == round(retina[COLUMN] * 255)
        assert (facet[0] == facet[0][0]).all(), "not grey"

    def test_facets_are_separated_by_dark_lines(self):
        layout = self._layout()
        image = layout.paint(np.ones(HEX_COLUMN_COUNT))
        square = image[
            layout.top : layout.top + layout.side, layout.left : layout.left + layout.side
        ]
        lines = layout.square_index == FacetLayout.DARK
        inside = np.zeros_like(lines)
        inside[layout.side // 4 : 3 * layout.side // 4, layout.side // 4 : 3 * layout.side // 4] = (
            True
        )
        assert (lines & inside).any(), "no lines between facets in the middle of the eye"
        assert not square[lines].any(), "the lines are not black"

    def test_outside_the_eye_is_black(self):
        image = self._layout().paint(np.ones(HEX_COLUMN_COUNT))
        assert not image[:, -1].any() and not image[-1, :].any()

    def test_a_new_frame_repaints_the_same_buffer(self):
        layout = self._layout()
        first = layout.paint(np.zeros(HEX_COLUMN_COUNT))
        second = layout.paint(np.ones(HEX_COLUMN_COUNT))
        assert first is second and second.any()

    def test_a_window_with_no_area_is_refused(self):
        with pytest.raises(ValueError):
            FacetLayout(0, 300, self.RASTER)

    def test_the_picture_relays_out_when_the_window_changes_size(self):
        picture = _picture()
        assert picture.fly_image(400, 300).shape == (300, 400, 3)
        assert picture.fly_image(640, 480).shape == (480, 640, 3)


class TestPlacement:
    @pytest.mark.parametrize("viewport_height", [480, 720, 960, 1440])
    def test_rects_hang_from_the_top_edge(self, viewport_height):
        """viewport over-reports the drawable area, so only the top edge can be trusted
        (see ViewerHUD); a rect anchored to the bottom lands off screen."""
        rect = place_overlay(viewport_height, left_px=12, top_px=30, width=100, height=50)
        assert viewport_height - (rect.bottom + rect.height) == 30
        assert (rect.left, rect.width, rect.height) == (12, 100, 50)

    def test_big_view_is_most_of_the_height_on_a_wide_window(self):
        assert big_view_side(4000, 1000, left_px=674, margin_px=12) == 700

    def test_big_view_never_runs_past_the_right_edge(self):
        side = big_view_side(1000, 1000, left_px=674, margin_px=12)
        assert 674 + side <= 1000 - 12

    def test_big_view_is_zero_when_there_is_no_room(self):
        assert big_view_side(600, 1000, left_px=674, margin_px=12) == 0


class _FakeResampler:
    def frame(self, frame: np.ndarray) -> np.ndarray:
        return np.full((1, 1, 1, HEX_COLUMN_COUNT), frame.mean() / 255.0, dtype=np.float32)


class _FakeEye:
    """Records what the view hands an eye; answers with zeros (an eye at rest)."""

    frame_shape = (96, 96, 3)
    frame_rate_hz = 50.0
    readout_names = READOUTS

    def __init__(self) -> None:
        self.resampler = _FakeResampler()
        self.frames: list[np.ndarray] = []
        self.sequences: list[tuple[np.ndarray, bool]] = []

    def encode(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return np.zeros(FEATURES, dtype=np.float32)

    def encode_sequence(self, frames: np.ndarray, *, reset: bool = True) -> np.ndarray:
        self.sequences.append((frames, reset))
        return np.zeros((len(frames), FEATURES), dtype=np.float32)


_SCENE = """
<mujoco>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.1" rgba="0.3 0.6 0.3 1"/>
    <geom type="box" pos="2 0 0.5" size="0.3 0.3 0.5" rgba="0.9 0.1 0.1 1"/>
    {camera}
  </worldbody>
</mujoco>
"""
_CAMERA = '<camera name="fly_head" pos="0 0 0.5" xyaxes="0 -1 0 0 0 1"/>'


def _model(camera: str = _CAMERA) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_SCENE.format(camera=camera))


class _Viewport:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height


def test_a_model_without_the_camera_is_refused_before_anything_renders():
    with pytest.raises(ValueError, match="fly_head"):
        FlyEyeView(_model(camera=""), _FakeEye())


@pytest.mark.render
class TestFlyEyeView:
    def _view(self, **kwargs) -> tuple[FlyEyeView, _FakeEye, mujoco.MjData]:
        model = _model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        eye = _FakeEye()
        return FlyEyeView(model, eye, **kwargs), eye, data

    def test_it_starts_settled_on_grey_for_one_second(self):
        view, eye, _ = self._view()
        try:
            [(frames, reset)] = eye.sequences
            assert reset and frames.shape == (50, 96, 96, 3)
        finally:
            view.close()

    def test_reset_settles_the_eye_on_the_view_in_front_of_it(self):
        view, eye, data = self._view()
        try:
            view.reset(data)
            frames, reset = eye.sequences[-1]
            assert reset and len(frames) == 50
            assert (frames == frames[0]).all(), "settling must hold one still frame"
            # Not exact: two renders of one state differ by a level in a handful of pixels
            # (measured: 14 of 27,648 values, off by 1) -- rasteriser jitter, not the view.
            np.testing.assert_allclose(frames[0], view.render_frame(data), atol=2)
            assert frames[0].std() > 0, "settled on a blank frame, not the scene"
        finally:
            view.close()

    def test_each_step_feeds_the_eye_exactly_one_camera_frame(self):
        view, eye, data = self._view()
        try:
            for _ in range(3):
                view.step(data)
            assert len(eye.frames) == 3
            assert all(
                frame.shape == (96, 96, 3) and frame.dtype == np.uint8 for frame in eye.frames
            )
            assert view.last_step_ms >= 0.0
        finally:
            view.close()

    def test_the_eye_is_stepped_even_while_the_view_is_off(self):
        """It is a dynamical system: one that skipped frames while hidden would be a
        different eye when shown again."""
        view, eye, data = self._view(mode=EyeViewMode.OFF)
        try:
            view.step(data)
            assert len(eye.frames) == 1
        finally:
            view.close()

    def test_corner_goes_under_the_telemetry_panel(self):
        view, _, data = self._view(below_px=HUD_HEIGHT, beside_px=HUD_WIDTH)
        try:
            view.step(data)
            rect, image = view.overlay(_Viewport(1706, 960))
            assert rect.left == 12
            assert 960 - (rect.bottom + rect.height) == 12 + HUD_HEIGHT + 12
            assert image.shape == (rect.height, rect.width, 3)
        finally:
            view.close()

    def test_big_goes_beside_the_telemetry_panel(self):
        view, _, data = self._view(mode=EyeViewMode.BIG, below_px=HUD_HEIGHT, beside_px=HUD_WIDTH)
        try:
            view.step(data)
            rect, image = view.overlay(_Viewport(1706, 960))
            assert rect.left == 12 + HUD_WIDTH + 12
            assert 960 - (rect.bottom + rect.height) == 12
            assert rect.width == rect.height and image.shape == (rect.height, rect.width, 3)
            assert rect.left + rect.width <= 1706
        finally:
            view.close()

    def test_without_a_hud_both_hug_the_corner(self):
        view, _, data = self._view()
        try:
            for mode in (EyeViewMode.CORNER, EyeViewMode.BIG):
                view.mode = mode
                rect, _ = view.overlay(_Viewport(1706, 960))
                assert rect.left == 12 and 960 - (rect.bottom + rect.height) == 12
        finally:
            view.close()

    def test_nothing_is_drawn_when_off_or_when_it_does_not_fit(self):
        view, _, _ = self._view()
        try:
            assert view.overlay(None) is None
            assert view.overlay(_Viewport(200, 150)) is None
            view.mode = EyeViewMode.OFF
            assert view.overlay(_Viewport(1706, 960)) is None
        finally:
            view.close()

    def test_the_fly_view_fills_the_whole_viewport(self):
        view, _, data = self._view(mode=EyeViewMode.FLY, below_px=HUD_HEIGHT, beside_px=HUD_WIDTH)
        try:
            view.step(data)
            rect, image = view.overlay(_Viewport(640, 360))
            assert (rect.left, rect.bottom, rect.width, rect.height) == (0, 0, 640, 360)
            assert image.shape == (360, 640, 3)
            assert view.covers_window
        finally:
            view.close()

    def test_only_the_fly_view_covers_the_window(self):
        view, _, _ = self._view()
        try:
            for mode in EyeViewMode:
                view.mode = mode
                assert view.covers_window is (mode is EyeViewMode.FLY)
        finally:
            view.close()

    def test_cycle_mode_steps_and_reports_the_new_mode(self):
        view, _, _ = self._view()
        try:
            assert view.key == EYE_VIEW_KEY
            assert view.cycle_mode() is EyeViewMode.BIG
            assert view.mode is EyeViewMode.BIG
        finally:
            view.close()


@pytest.mark.render
def test_the_real_eye_sees_a_moving_car_move():
    """The point of the view: parked is near black, driving lights it up.

    Measured when this was built: a parked car's mean motion read about 0.003 and the
    same car at speed about 0.035. The bar here is a conservative 3x.

    Skipped where CUDA is visible. ``import flyvis`` makes CUDA torch's default device and
    flyvis allocates on it internally whatever the eye is asked for, and the hex resampler
    then indexes CPU frames with a GPU tensor -- a separate bug that breaks every flyvis
    test on a GPU host. Run with ``CUDA_VISIBLE_DEVICES=-1`` until it is fixed.
    """
    pytest.importorskip("flyvis")
    import torch

    if torch.cuda.is_available():
        pytest.skip("flyvis on a CUDA host hits the resampler device bug; CUDA_VISIBLE_DEVICES=-1")
    from fly_driver.envs.car import CarConfig, CarDynamics, assemble_model_xml
    from fly_driver.envs.centerline import Centerline
    from fly_driver.envs.scene import SceneConfig
    from fly_driver.eyes.flyvis_eye import FlyvisEye, resolve_checkpoint_dir
    from fly_driver.interface import ControlVector

    try:
        resolve_checkpoint_dir()
    except FileNotFoundError:
        pytest.skip("run `flyvis download-pretrained` to enable this test")

    car = CarConfig()
    model = mujoco.MjModel.from_xml_string(
        assemble_model_xml(Centerline.load(), SceneConfig(), car)
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dynamics = CarDynamics(model, car)
    substeps = max(1, round(0.02 / model.opt.timestep))
    view = FlyEyeView(model, FlyvisEye())
    try:
        view.reset(data)
        strengths = []
        for step in range(200):
            throttle = 0.0 if step < 40 else 1.0
            dynamics.step(ControlVector(steer=0.0, throttle=throttle, brake=0.0), data, substeps)
            view.step(data)
            strengths.append(view.picture.motion_strength)
    finally:
        view.close()
    parked = float(np.mean(strengths[10:40]))
    moving = float(np.mean(strengths[120:200]))
    assert dynamics.speed_mps(data) > 20.0, "the car never got moving; the test proves nothing"
    assert moving > 3.0 * parked, f"moving {moving:.4f} vs parked {parked:.4f}"
