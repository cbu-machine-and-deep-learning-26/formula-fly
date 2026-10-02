"""Sizes the RQ1 eyes share, with no torch and no flyvis.

``FLYVIS_DEFAULT_FEATURE_DIM`` is the frozen flyvis T4/T5 readout:
eight direction-selective types (T4a–d, T5a–d) times 721 hexagonal columns.
Those two factors live on :class:`~fly_driver.eyes.flyvis_eye.FlyvisEye` and
:class:`~fly_driver.eyes.hex_resampler.HexResampler`, which import torch, so
this module repeats the product. ``tests/eyes`` asserts the three stay equal.
"""

FLYVIS_DEFAULT_FEATURE_DIM = 5768
