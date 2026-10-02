"""The brains package imports without torch and does not pull Brian2 into the loop."""

from __future__ import annotations

import sys

import fly_driver.brains as brains


def test_the_package_names_brian2_as_offline_and_does_not_import_it():
    assert brains.OFFLINE_SIMULATOR == "brian2"
    assert "brian2" not in sys.modules
    assert not hasattr(brains, "CentralComplexBrain")
