"""Shared numeric bounds for the live central-complex module and its blank control.

Both nets use the same leaky rate neuron, so a parameter-count comparison is about
wiring density rather than a different cell model.
"""

#: Default live width. A starting size, not a lock: neuron count is a sweep.
DEFAULT_NEURON_COUNT = 256

#: Smallest population that still has a compass ring, a steering cell, and a readout.
MIN_NEURON_COUNT = 16

#: Compass ring is at least this wide so ±1, ±2, and the opposite cell are distinct.
MIN_COMPASS_COUNT = 8

#: Each neuron reads this many eye features, clipped to the feature count.
DEFAULT_INPUT_IN_DEGREE = 16

#: Each steering and descending neuron reads this many partners inside the module.
DEFAULT_RECURRENT_IN_DEGREE = 16

#: Membrane time constant at init, seconds. A few frames at 50 Hz.
DEFAULT_TAU_SECONDS = 0.05

#: Trainable time constants stay inside this window so a step cannot explode or freeze.
MIN_TAU_SECONDS = 0.005
MAX_TAU_SECONDS = 0.200

#: Hard clamp on membrane voltage, after the leaky update.
VOLTAGE_LIMIT = 5.0

#: Initial random synapse scale is this over ``sqrt(in_degree)``.
WEIGHT_SCALE = 0.5

#: Synthetic ring: near neighbours excite, the opposite cell inhibits.
RING_EXCITATION_NEAR = 0.6
RING_EXCITATION_FAR = 0.3
RING_INHIBITION = -0.5
