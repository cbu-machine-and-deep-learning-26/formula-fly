"""GH-21 end to end: the real tethered body inside `EmbodiedDriveAgent` (GH-20).

Reuses the simple eye/policy doubles from `tests/test_drivers.py` rather than duplicating
them -- the point of the shared contract is that any real `Eye`/`Policy`/`Body` drops into
the same `EmbodiedDriveAgent`, so proving it here with `PooledLuminanceEye` and
`SteerTowardBrightPolicy` says the same thing it would with the real flyvis eye, without
needing torch in this environment.
"""

from __future__ import annotations

import pytest

from fly_driver.drivers import EmbodiedDriveAgent
from fly_driver.interface import Body, ControlVector

# The imports above are all lazy about flybody; the tests below are not, since they
# construct FlybodyWingBody/TrackballLegBody without an injected fake env, so they exercise
# real flybody physics. Skip the whole file together rather than guard each test.
pytest.importorskip("flybody")

from fly_driver.body import CombinedBody, FlybodyWingBody, TrackballLegBody  # noqa: E402
from tests.test_drivers import PooledLuminanceEye, SteerTowardBrightPolicy, _frame  # noqa: E402


def _embodied() -> EmbodiedDriveAgent:
    eye = PooledLuminanceEye()
    policy = SteerTowardBrightPolicy(eye.feature_dim)
    body = CombinedBody(FlybodyWingBody(), TrackballLegBody())
    return EmbodiedDriveAgent(eye, policy, body)


def test_the_real_body_satisfies_the_body_protocol() -> None:
    assert isinstance(CombinedBody(FlybodyWingBody(), TrackballLegBody()), Body)


def test_embodied_agent_drives_several_frames_without_raising() -> None:
    agent = _embodied()
    agent.reset(seed=0)

    for brighter in ("none", "right", "left", "right"):
        action = agent.act(_frame(brighter))
        assert isinstance(action, ControlVector)


def test_the_body_actually_changes_the_control_not_just_relays_it() -> None:
    """`PassthroughBody` would return the policy's intent unchanged; the real body must not."""
    agent = _embodied()
    agent.reset(seed=0)

    frame = _frame("right")
    intent = agent.policy.act(agent.eye.encode(frame))
    realised = agent.act(frame)

    assert (realised.steer, realised.throttle, realised.brake) != (
        intent.steer,
        intent.throttle,
        intent.brake,
    )
