"""Central complex and descending neurons, between the eye and the policy.

The closed-loop module is
:class:`fly_driver.brains.central_complex.CentralComplexBrain` (PyTorch). Import
that module by its path; importing this package does not import torch.

Brian2 is the offline simulator for the lesion map (GH-30) and sim-to-biology
(GH-31). It is not a dependency and it is not in the live loop. Neuron count is
a sweep parameter on the PyTorch module, not a fixed cap.
"""

OFFLINE_SIMULATOR = "brian2"

__all__ = ["OFFLINE_SIMULATOR"]
