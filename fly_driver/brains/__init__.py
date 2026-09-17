"""Brain stages. Identity is the Phase 0 default; LIF is a numerics stub."""

from fly_driver.brains.identity import IdentityBrain
from fly_driver.brains.lif import LeakyIntegrateFireLayer
from fly_driver.interface import Brain

__all__ = ["Brain", "IdentityBrain", "LeakyIntegrateFireLayer"]
