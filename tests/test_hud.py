"""Telemetry panel tests (GH-16).

The panel is a numpy image, so every element is checked by reading pixels back: no
display, no viewer, no toolkit. The viewer hand-off is tested against a stand-in handle.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fly_driver import hud
from fly_driver.hud import (
    HEIGHT,
    LED_COUNT,
    WIDTH,
    Telemetry,
    ViewerHUD,
    bar_fill,
    leds_lit,
    render,
    rpm_fraction,
    shift_light_on,
    steer_to_x,
)

LIMITER, SHIFT = 15000.0, 14100.0


def frame(**overrides) -> Telemetry:
    values = dict(speed_kmh=0.0, gear=1, rpm=0.0, throttle=0.0, brake=0.0, steer=0.0)
    values.update(overrides)
    return Telemetry(**values)


def draw(**overrides) -> np.ndarray:
    return render(frame(**overrides), limiter_rpm=LIMITER, shift_rpm=SHIFT)


def column_rows(img: np.ndarray, x: int, colour: tuple[int, int, int]) -> np.ndarray:
    """Row indices where column ``x`` is exactly ``colour``."""
    return np.flatnonzero((img[:, x] == np.array(colour, dtype=np.uint8)).all(axis=1))


def row_cols(img: np.ndarray, y: int, colour: tuple[int, int, int]) -> np.ndarray:
    return np.flatnonzero((img[y] == np.array(colour, dtype=np.uint8)).all(axis=1))


class TestRpm:
    def test_zero_at_rest(self):
        assert rpm_fraction(0.0, LIMITER) == 0.0

    def test_full_at_the_limiter(self):
        assert rpm_fraction(LIMITER, LIMITER) == 1.0

    def test_clips_past_the_limiter(self):
        assert rpm_fraction(20000.0, LIMITER) == 1.0

    def test_rejects_a_zero_limiter(self):
        with pytest.raises(ValueError):
            rpm_fraction(1000.0, 0.0)

    def test_shift_light_comes_on_at_the_shift_point_not_the_limiter(self):
        assert not shift_light_on(14000.0, SHIFT)
        assert shift_light_on(SHIFT, SHIFT)
        assert shift_light_on(14900.0, SHIFT)


class TestShiftLights:
    def test_none_lit_low_in_the_range(self):
        assert leds_lit(5000.0, LIMITER, SHIFT) == 0

    def test_all_lit_at_the_shift_point(self):
        assert leds_lit(SHIFT, LIMITER, SHIFT) == LED_COUNT

    def test_fill_in_one_by_one(self):
        start = hud.LED_START_FRACTION * LIMITER
        counts = [leds_lit(rpm, LIMITER, SHIFT) for rpm in np.linspace(start, SHIFT, 200)]
        assert counts == sorted(counts)
        assert set(counts) == set(range(LED_COUNT + 1))

    def test_rejects_a_shift_point_past_the_limiter(self):
        with pytest.raises(ValueError):
            leds_lit(1000.0, LIMITER, LIMITER + 1)


class TestPedalBars:
    def test_fill_is_the_value(self):
        assert bar_fill(0.35) == 0.35

    def test_clips_both_ends(self):
        assert bar_fill(-0.2) == 0.0
        assert bar_fill(1.7) == 1.0


class TestSteeringBar:
    """A horizontal bar from -1 to +1, the ControlVector's own range."""

    def test_full_left_is_the_left_edge(self):
        assert steer_to_x(-1.0, 100.0, 300.0) == 100.0

    def test_full_right_is_the_right_edge(self):
        assert steer_to_x(1.0, 100.0, 300.0) == 300.0

    def test_centred_is_the_middle(self):
        assert steer_to_x(0.0, 100.0, 300.0) == 200.0

    def test_half_is_halfway(self):
        assert steer_to_x(0.5, 100.0, 300.0) == 250.0

    def test_clips_beyond_the_range(self):
        assert steer_to_x(3.0, 100.0, 300.0) == 300.0
        assert steer_to_x(-3.0, 100.0, 300.0) == 100.0

    def test_rejects_an_inverted_bar(self):
        with pytest.raises(ValueError):
            steer_to_x(0.0, 300.0, 100.0)


class TestFont:
    def test_every_glyph_is_five_by_seven(self):
        for char, rows in hud._FONT.items():
            assert len(rows) == hud.GLYPH_H, char
            assert all(len(row) == hud.GLYPH_W for row in rows), char
            assert all(set(row) <= {"#", "."} for row in rows), char

    def test_digits_are_distinct(self):
        digits = [hud._FONT[str(d)] for d in range(10)]
        assert len(set(digits)) == 10

    def test_everything_the_panel_prints_has_a_glyph(self):
        for text in (
            "0123456789",
            "KM/H",
            "RPM",
            "GEAR",
            "STEER",
            "SUSP",
            "T",
            "B",
            "-1",
            "+1",
            "FL",
            "FR",
            "RL",
            "RR",
        ):
            for char in text:
                assert char in hud._FONT, char


class TestRender:
    def test_shape_and_dtype(self):
        img = draw()
        assert img.shape == (HEIGHT, WIDTH, 3)
        assert img.dtype == np.uint8

    def test_rejects_a_shift_point_past_the_limiter(self):
        with pytest.raises(ValueError):
            render(frame(), limiter_rpm=LIMITER, shift_rpm=LIMITER + 1)

    def test_throttle_bar_height_follows_the_pedal(self):
        x = sum(hud._THROTTLE_X) // 2
        top, bottom = hud._PEDAL_Y
        empty = column_rows(draw(throttle=0.0), x, hud._THROTTLE)
        half = column_rows(draw(throttle=0.5), x, hud._THROTTLE)
        full = column_rows(draw(throttle=1.0), x, hud._THROTTLE)
        assert empty.size == 0
        assert half.size == pytest.approx((bottom - top) / 2, abs=1)
        assert full.size == bottom - top
        assert full.max() == bottom - 1  # fills upward from the bottom

    def test_brake_bar_is_red_and_independent_of_throttle(self):
        x = sum(hud._BRAKE_X) // 2
        assert column_rows(draw(brake=1.0), x, hud._BRAKE).size == hud._PEDAL_Y[1] - hud._PEDAL_Y[0]
        assert column_rows(draw(throttle=1.0), x, hud._BRAKE).size == 0

    def test_steering_marker_runs_from_left_edge_to_right_edge(self):
        x0, y0, x1, y1 = hud._STEER_BAR
        y = (y0 + y1) // 2
        left = row_cols(draw(steer=-1.0), y, hud._STEER)
        centre = row_cols(draw(steer=0.0), y, hud._STEER)
        right = row_cols(draw(steer=1.0), y, hud._STEER)
        assert left.min() == x0 - hud._STEER_MARKER_HALF_WIDTH
        assert right.max() == x1 + hud._STEER_MARKER_HALF_WIDTH - 1
        assert centre.mean() == pytest.approx((x0 + x1) / 2, abs=1)

    def test_rpm_bar_fills_with_engine_speed(self):
        x0, y0, x1, y1 = hud._RPM_BAR
        y = (y0 + y1) // 2
        assert row_cols(draw(rpm=0.0), y, hud._RPM).size == 0
        half = row_cols(draw(rpm=LIMITER / 2), y, hud._RPM)
        assert half.size == pytest.approx((x1 - x0) / 2, abs=1)
        assert half.min() == x0

    def test_shift_lights_count_up_then_all_go_blue(self):
        y = sum(hud._LED_Y) // 2
        xs = [hud._MARGIN + i * hud._LED_PITCH + hud._LED_WIDTH // 2 for i in range(LED_COUNT)]

        def lit(img):
            return [tuple(img[y, x]) not in hud._LED_OFF for x in xs]

        assert lit(draw(rpm=5000.0)) == [False] * LED_COUNT
        assert lit(draw(rpm=13000.0)).count(True) == leds_lit(13000.0, LIMITER, SHIFT)
        shifting = draw(rpm=SHIFT)
        assert all(tuple(shifting[y, x]) == hud._LED_SHIFT for x in xs)

    def test_suspension_bars_go_up_for_compression_and_down_for_extension(self):
        top, bottom = hud._SUSP_Y
        mid = (top + bottom) // 2
        img = draw(suspension=(1.0, -1.0, 0.0, 0.5))
        fl = column_rows(img, sum(hud._SUSP_X["fl"]) // 2, hud._SUSP_COMPRESS)
        fr = column_rows(img, sum(hud._SUSP_X["fr"]) // 2, hud._SUSP_EXTEND)
        rl_up = column_rows(img, sum(hud._SUSP_X["rl"]) // 2, hud._SUSP_COMPRESS)
        rr = column_rows(img, sum(hud._SUSP_X["rr"]) // 2, hud._SUSP_COMPRESS)
        assert fl.min() == top and fl.max() < mid
        assert fr.min() > mid and fr.max() == bottom - 1
        assert rl_up.size == 0
        assert rr.size == pytest.approx((mid - top) / 2, abs=2)

    def test_speed_digits_change_the_picture(self):
        assert not np.array_equal(draw(speed_kmh=100.0), draw(speed_kmh=200.0))
        assert not np.array_equal(draw(gear=3), draw(gear=4))

    def test_out_of_range_inputs_still_draw(self):
        img = draw(throttle=5.0, brake=-1.0, steer=9.0, rpm=1e6, suspension=(3.0, -3.0, 0.0, 0.0))
        assert img.shape == (HEIGHT, WIDTH, 3)


class _FakeViewer:
    def __init__(self, width: int, height: int) -> None:
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        self.images: list = []
        self.texts: list = []

    def set_images(self, viewports_images) -> None:
        self.images.append(viewports_images)

    def set_texts(self, texts) -> None:
        self.texts.append(texts)


class TestViewerHUD:
    def test_rejects_a_shift_point_past_the_limiter(self):
        with pytest.raises(ValueError):
            ViewerHUD(_FakeViewer(1280, 720), limiter_rpm=LIMITER, shift_rpm=LIMITER + 1)

    def test_hands_the_viewer_one_panel_in_the_top_left(self):
        viewer = _FakeViewer(1280, 720)
        panel = ViewerHUD(viewer, limiter_rpm=LIMITER, shift_rpm=SHIFT, margin_px=12)
        panel.update(frame(speed_kmh=100.0))
        [(rect, image)] = viewer.images[-1]
        assert (rect.left, rect.width, rect.height) == (12, WIDTH, HEIGHT)
        assert rect.bottom == 720 - HEIGHT - 12
        assert image.shape == (rect.height, rect.width, 3)

    def test_the_panel_hugs_the_top_whatever_the_window_height(self):
        """The bug this replaced: anchoring to the bottom put the panel below the visible
        area of the window, where it drew every frame and was never seen. viewport
        over-reports the drawable height, so only the top edge can be trusted."""
        for height in (480, 720, 960, 1440):
            viewer = _FakeViewer(1280, height)
            ViewerHUD(viewer, limiter_rpm=LIMITER, shift_rpm=SHIFT, margin_px=12).update(frame())
            [(rect, _)] = viewer.images[-1]
            gap_above = height - (rect.bottom + rect.height)
            assert gap_above == 12, f"panel drifted from the top at height {height}"

    def test_it_also_sets_a_text_overlay(self):
        """Do not delete this as dead code. MuJoCo's passive viewer only runs its overlay
        pass when a text overlay is set: without it set_images() is accepted every frame,
        raises nothing, and draws nothing, which is how this panel first shipped
        invisible. Proven by A/B capture of the real viewer."""
        viewer = _FakeViewer(1280, 720)
        ViewerHUD(viewer, limiter_rpm=LIMITER, shift_rpm=SHIFT).update(frame())
        assert viewer.texts, "no text overlay set; the image will not be drawn"

    def test_the_text_overlay_is_empty_so_only_the_panel_shows(self):
        viewer = _FakeViewer(1280, 720)
        ViewerHUD(viewer, limiter_rpm=LIMITER, shift_rpm=SHIFT).update(frame())
        _, _, left, right = viewer.texts[-1]
        assert (left, right) == ("", "")

    def test_skips_a_window_too_small_for_the_panel(self):
        viewer = _FakeViewer(300, 100)
        ViewerHUD(viewer, limiter_rpm=LIMITER, shift_rpm=SHIFT).update(frame())
        assert viewer.images == []
