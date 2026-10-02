"""Collection rules that depend on what is installed on this machine.

The eye (`fly_driver/eyes/`) imports torch at module scope, and torch is deliberately not a
dependency of this project: flyvis needs Python <3.13 and platform-specific CUDA wheels, so
it installs into its own virtualenv from `requirements-flyvis.txt` (see
`docs/running-the-stacks.md`). Several of `tests/eyes/` import it without a guard, and
`scripts/flyvis_eye_demo.py` cannot be imported at all without it -- so on a machine with
only the base install, a bare ``pytest -q`` is four collection errors and a failure rather
than a test result.

Skipping them here rather than passing ``--ignore=tests/eyes`` on the command line matters:
``--ignore`` hides those tests from everyone, including the machines that *do* have torch and
are the only ones that can run them. This checks first, so they run wherever they can.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator

import pytest

#: Paths relative to this directory, skipped only when torch is missing. Empty otherwise, so
#: nothing is hidden on a machine that can actually run them.
_NEEDS_TORCH = [
    "eyes/*",
    "analysis/*",
    "scripts/test_flyvis_eye_demo.py",
    "scripts/test_flyvis_eye_live.py",
]

collect_ignore_glob: list[str] = [] if importlib.util.find_spec("torch") else _NEEDS_TORCH


@pytest.fixture
def cpu_default_device() -> Iterator[None]:
    """Run the test with the CPU as torch's default device, then restore the previous one.

    ``import flyvis`` calls ``torch.set_default_device("cuda")`` on any GPU machine (ROCm
    included) and never undoes it, so the first test that imports flyvis changes the default
    for every later test in the same process. Tests about numerics rather than devices opt out
    of that with ``pytestmark = pytest.mark.usefixtures("cpu_default_device")``; the GPU paths
    have tests of their own.
    """
    import torch

    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(previous)
