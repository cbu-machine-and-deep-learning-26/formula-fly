"""Policies and baseline agents.

:class:`~fly_driver.policies.mlp_policy.MlpPolicy`, the trainable readout, imports torch and
is deliberately not re-exported here: take it by its full module path so this package stays
importable in the base install.
"""

from fly_driver.policies.baselines import ConstantAgent, RandomAgent

__all__ = ["ConstantAgent", "RandomAgent"]
