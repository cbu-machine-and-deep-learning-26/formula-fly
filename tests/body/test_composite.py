"""Tests for `CombinedBody` (GH-21). No flybody needed: both sub-bodies are fakes."""

from __future__ import annotations

import fly_driver.body
from fly_driver.body.composite import CombinedBody
from fly_driver.interface import ControlVector


class _FakeBody:
    """A `Body` double that returns a fixed control and records calls."""

    def __init__(self, result: ControlVector) -> None:
        self.result = result
        self.reset_count = 0
        self.last_intent: ControlVector | None = None

    def reset(self) -> None:
        self.reset_count += 1

    def actuate(self, intent: ControlVector) -> ControlVector:
        self.last_intent = intent
        return self.result


def test_package_exports_combined_body() -> None:
    assert fly_driver.body.CombinedBody is CombinedBody


def test_reset_resets_both_bodies() -> None:
    steer_body = _FakeBody(ControlVector.neutral())
    throttle_body = _FakeBody(ControlVector.neutral())
    combined = CombinedBody(steer_body, throttle_body)

    combined.reset()

    assert steer_body.reset_count == 1
    assert throttle_body.reset_count == 1


def test_both_bodies_see_the_same_intent() -> None:
    steer_body = _FakeBody(ControlVector.neutral())
    throttle_body = _FakeBody(ControlVector.neutral())
    combined = CombinedBody(steer_body, throttle_body)
    combined.reset()

    intent = ControlVector(steer=0.5, throttle=0.4, brake=0.1)
    combined.actuate(intent)

    assert steer_body.last_intent is intent
    assert throttle_body.last_intent is intent


def test_steer_comes_from_the_steer_body_only() -> None:
    steer_body = _FakeBody(ControlVector(steer=0.9, throttle=0.0, brake=0.0))
    throttle_body = _FakeBody(ControlVector(steer=-0.9, throttle=0.6, brake=0.2))
    combined = CombinedBody(steer_body, throttle_body)
    combined.reset()

    realised = combined.actuate(ControlVector.neutral())

    assert realised.steer == 0.9


def test_throttle_and_brake_come_from_the_throttle_body_only() -> None:
    steer_body = _FakeBody(ControlVector(steer=0.9, throttle=0.7, brake=0.3))
    throttle_body = _FakeBody(ControlVector(steer=-0.9, throttle=0.6, brake=0.2))
    combined = CombinedBody(steer_body, throttle_body)
    combined.reset()

    realised = combined.actuate(ControlVector.neutral())

    assert realised.throttle == 0.6
    assert realised.brake == 0.2
