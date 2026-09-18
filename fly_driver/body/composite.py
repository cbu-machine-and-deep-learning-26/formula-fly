"""Compose two single-purpose bodies into one `Body` (GH-21).

`FlybodyWingBody` and `TrackballLegBody` each drive one physically separate flybody
environment -- a real tethered rig is one or the other, not both in the same MuJoCo scene,
since flybody wires a wing pattern generator into the flight task and nothing equivalent
into the walking one (see both modules' docstrings). `CombinedBody` is what issue #21 asks
for regardless: wings steer, trackball drives throttle and brake, together.
"""

from __future__ import annotations

from fly_driver.interface import Body, ControlVector

__all__ = ["CombinedBody"]


class CombinedBody:
    """Wings steer, a trackball throttles: two `Body` stages, one `Body` to the pipeline.

    Args:
        steer_body: Actuated with the same intent every step; only its ``steer`` is kept.
        throttle_body: Actuated with the same intent every step; its ``throttle`` and
            ``brake`` are kept, its ``steer`` discarded.
    """

    def __init__(self, steer_body: Body, throttle_body: Body) -> None:
        self.steer_body = steer_body
        self.throttle_body = throttle_body

    def reset(self) -> None:
        """Reset both underlying bodies."""
        self.steer_body.reset()
        self.throttle_body.reset()

    def actuate(self, intent: ControlVector) -> ControlVector:
        """Actuate both bodies with ``intent``; steer from one, throttle/brake from the other."""
        steer_result = self.steer_body.actuate(intent)
        throttle_result = self.throttle_body.actuate(intent)
        return ControlVector.clipped(
            steer=steer_result.steer,
            throttle=throttle_result.throttle,
            brake=throttle_result.brake,
        )
