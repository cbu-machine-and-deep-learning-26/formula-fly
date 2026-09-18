"""Whole-brain models between the eye and the policy.

Nothing heavy is imported here. The Brian2 and PyTorch backends each pull their
simulator in lazily, and the connectome loader pulls in pandas the same way, so
``import fly_driver.brains`` works on the base install with neither present.

That is a deliberate difference from ``fly_driver.eyes``, whose ``__init__`` imports
torch at module scope -- which is why four of its test modules cannot be collected
without it and ``tests/conftest.py`` has to skip them.
"""

from fly_driver.brains.benchmark import (
    BIOLOGICAL_MS_PER_FRAME,
    BenchmarkConfig,
    NeuronParameters,
    ThroughputResult,
)
from fly_driver.brains.connectome import (
    CONNECTOME_DIR_ENV,
    ConnectomeFiles,
    ConnectomeNotFoundError,
    Subnetwork,
    find_files,
    load_subnetwork,
    resolve_connectome_dir,
    subnetwork_from_arrays,
)

__all__ = [
    "BIOLOGICAL_MS_PER_FRAME",
    "CONNECTOME_DIR_ENV",
    "BenchmarkConfig",
    "ConnectomeFiles",
    "ConnectomeNotFoundError",
    "NeuronParameters",
    "Subnetwork",
    "ThroughputResult",
    "find_files",
    "load_subnetwork",
    "resolve_connectome_dir",
    "subnetwork_from_arrays",
]
