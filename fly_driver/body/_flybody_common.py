"""What `flybody_wing.py` and `flybody_legs.py` both need: shared, not duplicated.

Two copies of the same lazy import and the same tab-joined action-name lookup would
drift apart the first time either needed a fix. Private (leading underscore, no
``__all__``) because nothing outside :mod:`fly_driver.body` should import from here
directly -- the public names are re-exported from the two body modules themselves.
"""

from __future__ import annotations

from typing import Any


class FlybodyNotInstalledError(ImportError):
    """Raised when the optional flybody stack is needed but not importable."""


def _import_flybody() -> Any:
    try:
        import flybody.fly_envs as fly_envs
    except ImportError as error:
        raise FlybodyNotInstalledError(
            "This body needs the optional flybody stack. Install it with "
            "`python -m pip install "
            '"flybody @ git+https://github.com/TuragaLab/flybody.git@'
            'd015e9bfe441bd90ae431bac24c55cb74bdbce26"` '
            "in its own virtualenv -- see docs/running-the-stacks.md."
        ) from error
    return fly_envs


def _action_index_map(action_name: str) -> dict[str, int]:
    """``{component name: index}`` from a dm_control action_spec's tab-joined name."""
    names = action_name.split("\t")
    return {name: index for index, name in enumerate(names)}
