"""The tethered fly body: an intended control in, the cockpit's reading out (GH-21)."""

from fly_driver.body.composite import CombinedBody
from fly_driver.body.flybody_legs import TrackballLegBody
from fly_driver.body.flybody_wing import (
    FlybodyNotInstalledError,
    FlybodyWingBody,
)

__all__ = [
    "CombinedBody",
    "FlybodyNotInstalledError",
    "FlybodyWingBody",
    "TrackballLegBody",
]
