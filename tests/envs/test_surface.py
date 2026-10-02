"""Grip by surface (GH-16).

The whole feature is a silent one: if it stops working the car simply grips everywhere,
which is what it did before and which looks entirely normal from the outside. So the tests
check the mechanism rather than the vibe -- that the ground plane cannot override the tyre,
that grip really is taken away past the kerb, and that two wheels off gives a split the car
can spin on.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fly_driver.envs.car import CarConfig, assemble_model_xml
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import SceneConfig
from fly_driver.envs.surface import WHEEL_NAMES, SurfaceGrip

CAR = CarConfig()


@pytest.fixture(scope="module")
def straight() -> Centerline:
    """A straight 10 m wide road along +x, so lateral offset is just y."""
    return Centerline(
        points=[(0.0, 0.0), (400.0, 0.0), (800.0, 0.0)],
        half_width_right=[5.0] * 3,
        half_width_left=[5.0] * 3,
    )


@pytest.fixture(scope="module")
def model(straight) -> mujoco.MjModel:
    scene = SceneConfig(mesh_spacing_m=100.0, kerb_width_m=0.0)
    return mujoco.MjModel.from_xml_string(assemble_model_xml(straight, scene, CAR))


def _grip(model, straight, **kwargs) -> SurfaceGrip:
    return SurfaceGrip(model, straight, car=CAR, **kwargs)


class TestTheGroundPlaneCannotOverrideTheTyre:
    """MuJoCo takes the element-wise maximum of the two geoms' friction, so a ground plane
    set anywhere near the tyre's value becomes a floor under grip and no amount of scaling
    the wheel down can take grip away. This is the reason the plane is set to 0.05."""

    def test_the_plane_is_slippier_than_the_softest_tyre(self, model):
        ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        wheels = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{name}_geom")
            for name in WHEEL_NAMES
        ]
        softest = min(model.geom_friction[wheel][0] for wheel in wheels)
        assert model.geom_friction[ground][0] < softest * SceneConfig().grass_friction_scale

    def test_a_scaled_down_tyre_still_wins_the_maximum(self, model, straight):
        """The property that matters, stated as MuJoCo resolves it rather than as intent."""
        ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        grip = _grip(model, straight, grass_friction_scale=0.35)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        grip.reset()
        for name in WHEEL_NAMES:
            wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{name}_geom")
            on_grass = model.geom_friction[wheel][0] * 0.35
            assert on_grass > model.geom_friction[ground][0], (
                "the plane would win the max and the grass would grip like asphalt"
            )


class TestWhereTheGrassStarts:
    def test_a_wheel_in_the_middle_of_the_road_is_on_asphalt(self, model, straight):
        assert _grip(model, straight).grass_fraction_at(100.0, 0.0, 0.4) == 0.0

    def test_a_wheel_well_past_the_edge_is_fully_on_grass(self, model, straight):
        assert _grip(model, straight).grass_fraction_at(100.0, 20.0, 0.4) == 1.0

    def test_it_is_blended_across_the_wheel_not_switched(self, model, straight):
        """A hard step would put a discontinuity in the contact forces under the most
        loaded wheel, which MuJoCo resolves as a jolt."""
        grip = _grip(model, straight)
        width = 0.4
        straddling = grip.grass_fraction_at(100.0, 5.0, width)
        assert 0.0 < straddling < 1.0
        assert straddling == pytest.approx(0.5, abs=0.05), "the edge should be the halfway point"

    def test_the_blend_is_monotonic(self, model, straight):
        grip = _grip(model, straight)
        fractions = [grip.grass_fraction_at(100.0, y, 0.4) for y in np.linspace(4.0, 6.0, 25)]
        assert fractions == sorted(fractions)

    def test_both_sides_of_the_road_behave_the_same(self, model, straight):
        grip = _grip(model, straight)
        assert grip.grass_fraction_at(100.0, 6.0, 0.4) == pytest.approx(
            grip.grass_fraction_at(100.0, -6.0, 0.4)
        )

    def test_the_kerb_grips_like_the_circuit(self, model, straight):
        """Same line the track-limits rule uses, so what costs a driver seconds and what
        costs them grip are the same edge rather than two that nearly agree."""
        on_kerb = _grip(model, straight, kerb_width_m=2.0).grass_fraction_at(100.0, 5.5, 0.4)
        assert on_kerb == 0.0

    def test_a_wider_kerb_pushes_the_grass_further_out(self, model, straight):
        narrow = _grip(model, straight, kerb_width_m=0.0).grass_fraction_at(100.0, 6.0, 0.4)
        wide = _grip(model, straight, kerb_width_m=3.0).grass_fraction_at(100.0, 6.0, 0.4)
        assert narrow > wide


class TestApplyingIt:
    @pytest.fixture(autouse=True)
    def _asphalt(self, model, straight):
        """The model fixture is shared and this class mutates it in place."""
        _grip(model, straight).reset()

    def _place(self, model, y: float) -> mujoco.MjData:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        data.qpos[0], data.qpos[1] = 100.0, y
        mujoco.mj_forward(model, data)
        return data

    def test_on_the_road_every_wheel_keeps_full_grip(self, model, straight):
        grip = _grip(model, straight)
        grip.apply(self._place(model, 0.0))
        assert grip.scales == (1.0, 1.0, 1.0, 1.0)
        assert not grip.is_any_wheel_on_grass

    def test_off_the_road_every_wheel_loses_it(self, model, straight):
        grip = _grip(model, straight, grass_friction_scale=0.35)
        grip.apply(self._place(model, 40.0))
        assert grip.scales == pytest.approx((0.35, 0.35, 0.35, 0.35))
        assert grip.is_any_wheel_on_grass

    def test_two_wheels_off_gives_a_split_the_car_can_spin_on(self, model, straight):
        """The reason grip is per wheel rather than per car. A single car-wide factor
        could not produce this, and the asymmetry is what actually spins the car."""
        grip = _grip(model, straight, grass_friction_scale=0.35)
        # Centred on the left-hand edge: the left wheels are still on the asphalt while
        # the right pair is fully into the grass.
        grip.apply(self._place(model, 5.0))
        scales = grip.scales
        assert min(scales) == pytest.approx(0.35), f"expected two wheels off, got {scales}"
        assert max(scales) == 1.0, f"expected two wheels still on, got {scales}"

    def test_the_friction_written_to_the_model_matches_the_scale(self, model, straight):
        grip = _grip(model, straight, grass_friction_scale=0.5)
        baseline = [
            CAR.wheel_friction[0] if name.startswith("f") else CAR.wheel_friction_rear[0]
            for name in WHEEL_NAMES
        ]
        grip.apply(self._place(model, 40.0))
        for name, was in zip(WHEEL_NAMES, baseline, strict=True):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{name}_geom")
            assert model.geom_friction[geom][0] == pytest.approx(was * 0.5)
        grip.reset()

    def test_reset_restores_asphalt_grip(self, model, straight):
        """The model is mutated in place, so a car respawned on the grid would otherwise
        keep whatever grip it had when it stopped -- off in the grass, usually."""
        grip = _grip(model, straight, grass_friction_scale=0.35)
        grip.apply(self._place(model, 40.0))
        grip.reset()
        assert grip.scales == (1.0, 1.0, 1.0, 1.0)
        for name in WHEEL_NAMES:
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{name}_geom")
            expected = CAR.wheel_friction[0] if name.startswith("f") else CAR.wheel_friction_rear[0]
            assert model.geom_friction[geom][0] == pytest.approx(expected)

    def test_grip_comes_back_when_the_car_rejoins(self, model, straight):
        """Applying is not one-way. A car that runs wide and recovers must get its grip
        back, or one mistake would quietly ruin the rest of the lap."""
        grip = _grip(model, straight, grass_friction_scale=0.35)
        grip.apply(self._place(model, 40.0))
        grip.apply(self._place(model, 0.0))
        assert grip.scales == (1.0, 1.0, 1.0, 1.0)

    def test_a_scale_of_one_leaves_the_car_exactly_as_it_was(self, model, straight):
        """The escape hatch back to the old behaviour, where grass gripped like asphalt."""
        grip = _grip(model, straight, grass_friction_scale=1.0)
        grip.apply(self._place(model, 40.0))
        assert grip.scales == (1.0, 1.0, 1.0, 1.0)


class TestValidation:
    @pytest.mark.parametrize("scale", [0.0, -0.1, 1.5])
    def test_the_scale_must_be_a_usable_fraction(self, model, straight, scale):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _grip(model, straight, grass_friction_scale=scale)

    def test_a_negative_kerb_is_refused(self, model, straight):
        with pytest.raises(ValueError, match="non-negative"):
            _grip(model, straight, kerb_width_m=-1.0)

    def test_a_model_without_wheels_is_refused(self, straight):
        bare = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="plane" size="0 0 1"/></worldbody></mujoco>'
        )
        with pytest.raises(ValueError, match="no wheel named"):
            SurfaceGrip(bare, straight)


class TestTheBaselineCannotDrift:
    """SurfaceGrip mutates geom_friction in place, so taking the baseline from the model
    would let it compound: build one while the car is on the grass and the reduced value
    becomes the new asphalt. Nothing raises -- grip just quietly decays."""

    def test_a_second_grip_built_on_a_scaled_model_agrees_with_the_first(self, model, straight):
        first = _grip(model, straight, grass_friction_scale=0.35)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        data.qpos[0], data.qpos[1] = 100.0, 40.0
        mujoco.mj_forward(model, data)
        first.apply(data)  # the model now holds grass friction

        second = _grip(model, straight, grass_friction_scale=0.35)
        second.reset()
        for name in WHEEL_NAMES:
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{name}_geom")
            expected = CAR.wheel_friction[0] if name.startswith("f") else CAR.wheel_friction_rear[0]
            assert model.geom_friction[geom][0] == pytest.approx(expected)

    def test_applying_repeatedly_does_not_compound(self, model, straight):
        grip = _grip(model, straight, grass_friction_scale=0.35)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        data.qpos[0], data.qpos[1] = 100.0, 40.0
        mujoco.mj_forward(model, data)
        for _ in range(20):
            grip.apply(data)
        assert grip.scales == pytest.approx((0.35, 0.35, 0.35, 0.35))
        grip.reset()
