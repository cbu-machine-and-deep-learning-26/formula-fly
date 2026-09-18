"""Loading FlyWire connectivity, and carving subsets out of it (GH-23).

The whole-brain model of Shiu et al. is a list of neurons and a list of weighted,
directed synapses between them. This module is the part of that which needs no
simulator: find the data, load it, and select a subnetwork. Everything that actually
integrates neurons lives next door in :mod:`fly_driver.brains.shiu` and
:mod:`fly_driver.brains.torch_lif`.

Keeping the two apart matters more than it looks. The measurement this ticket exists
for is "which backend, and how big a network fits 20 ms of biology into the frame
budget", and that question is only meaningful if both backends are handed *the same*
network. They are, because they both take a :class:`Subnetwork` from here.

**The data is not in this repository and must never be.** FlyWire connectivity is
370 MB of parquet; it is downloaded once, kept outside the tree, and pointed at with
``FLY_CONNECTOME_DIR``. See ``docs/running-the-stacks.md``.

pandas and pyarrow are imported lazily, so ``import fly_driver.brains`` works on the
base install. That is deliberate: importing the eye package pulls torch in at module
scope, which is why four of its test modules cannot be collected without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import pandas as pd

__all__ = [
    "CONNECTOME_DIR_ENV",
    "ConnectomeFiles",
    "ConnectomeNotFoundError",
    "Subnetwork",
    "load_subnetwork",
    "resolve_connectome_dir",
]

#: Environment variable naming the directory holding the FlyWire parquet and CSV.
#: Follows the eye's ``FLYVIS_ROOT_DIR`` convention rather than inventing a new one.
CONNECTOME_DIR_ENV = "FLY_CONNECTOME_DIR"

#: FlyWire release -> (completeness CSV, connectivity parquet), named as the upstream
#: repository ships them. v630 is what the paper used; v783 is the current public one.
_RELEASES: dict[str, tuple[str, str]] = {
    "630": (
        "2023_03_23_completeness_630_final.csv",
        "2023_03_23_connectivity_630_final.parquet",
    ),
    "783": ("Completeness_783.csv", "Connectivity_783.parquet"),
}

#: Column in the connectivity table holding the signed synapse count. Upstream's own
#: name, kept verbatim -- AGENTS.md 14.1 says match the domain vocabulary rather than
#: renaming it on the way in.
_WEIGHT_COLUMN = "Excitatory x Connectivity"
_PRE_COLUMN = "Presynaptic_Index"
_POST_COLUMN = "Postsynaptic_Index"


class ConnectomeNotFoundError(RuntimeError):
    """Raised when the FlyWire data is not where we were told to look for it."""


@dataclass(frozen=True)
class ConnectomeFiles:
    """Where one FlyWire release lives on this machine.

    Args:
        release: FlyWire version, ``"630"`` or ``"783"``.
        completeness: CSV listing every neuron, indexed by ``root_id``.
        connectivity: Parquet of weighted directed synapses.
    """

    release: str
    completeness: Path
    connectivity: Path


@dataclass(frozen=True)
class Subnetwork:
    """A network ready to hand to a simulator backend.

    Args:
        root_ids: FlyWire ``root_id`` per neuron, in the order the backends index
            them, so a spike from index ``i`` can be named afterwards.
        pre: Presynaptic neuron index per synapse.
        post: Postsynaptic neuron index per synapse.
        weight: Signed synapse count per synapse; negative is inhibitory. Scaling it
            into millivolts is the backend's job, not this module's.
        release: Which FlyWire release it came from.
    """

    root_ids: npt.NDArray[np.int64]
    pre: npt.NDArray[np.int64]
    post: npt.NDArray[np.int64]
    weight: npt.NDArray[np.float64]
    release: str

    @property
    def num_neurons(self) -> int:
        """How many neurons are in this subnetwork."""
        return int(len(self.root_ids))

    @property
    def num_synapses(self) -> int:
        """How many synapses are in this subnetwork."""
        return int(len(self.pre))

    def __repr__(self) -> str:
        return (
            f"Subnetwork(release={self.release!r}, "
            f"neurons={self.num_neurons:,}, synapses={self.num_synapses:,})"
        )


def resolve_connectome_dir(directory: str | os.PathLike[str] | None = None) -> Path:
    """Find the directory holding the FlyWire data.

    Args:
        directory: An explicit path. ``None`` reads ``FLY_CONNECTOME_DIR``.

    Raises:
        ConnectomeNotFoundError: If neither is set, or the path does not exist.
    """
    raw = directory if directory is not None else os.environ.get(CONNECTOME_DIR_ENV)
    if raw is None:
        raise ConnectomeNotFoundError(
            f"set {CONNECTOME_DIR_ENV} to the directory holding the FlyWire "
            "connectivity parquet, or pass one explicitly. The data is a 370 MB "
            "download and deliberately lives outside this repository -- see "
            "docs/running-the-stacks.md."
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise ConnectomeNotFoundError(f"{path} is not a directory")
    return path


def find_files(
    directory: str | os.PathLike[str] | None = None, release: str = "630"
) -> ConnectomeFiles:
    """Locate one release's two files.

    Args:
        directory: Where to look; ``None`` reads ``FLY_CONNECTOME_DIR``.
        release: ``"630"`` (the paper's) or ``"783"`` (current public).

    Raises:
        ValueError: On an unknown release.
        ConnectomeNotFoundError: If either file is missing.
    """
    if release not in _RELEASES:
        known = ", ".join(sorted(_RELEASES))
        raise ValueError(f"unknown release {release!r}; known releases are {known}")
    root = resolve_connectome_dir(directory)
    completeness_name, connectivity_name = _RELEASES[release]
    files = ConnectomeFiles(
        release=release,
        completeness=root / completeness_name,
        connectivity=root / connectivity_name,
    )
    for path in (files.completeness, files.connectivity):
        if not path.is_file():
            raise ConnectomeNotFoundError(
                f"{path} is missing. Release {release} needs both "
                f"{completeness_name} and {connectivity_name}."
            )
    return files


def _read_tables(files: ConnectomeFiles) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two tables, importing pandas only when we actually need it."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on the local install
        raise ImportError(
            "reading the connectome needs pandas and pyarrow, which are not part of "
            "the base install. See docs/running-the-stacks.md for the brain "
            "environment."
        ) from exc
    return (
        pd.read_csv(files.completeness, index_col=0),
        pd.read_parquet(files.connectivity),
    )


def load_subnetwork(
    directory: str | os.PathLike[str] | None = None,
    *,
    release: str = "630",
    num_neurons: int | None = None,
) -> Subnetwork:
    """Load a release and optionally keep only the first ``num_neurons`` of it.

    The subset is contiguous in the upstream index rather than chosen by cell type,
    because the upstream repository ships **no cell-type annotations** -- only
    ``root_id``s and a completeness flag. Naming the central complex needs FlyWire's
    own annotation table, a separate download (see ``docs/running-the-stacks.md``).

    That is fine for the question this ticket asks. The cost of a timestep depends on
    how many neurons and synapses are being integrated, not on which ones they are, so
    a size-matched subset measures the budget just as well as the real one -- and it
    turns "is the central complex fast enough?" into the more useful "how large a
    network fits the frame budget?".

    Args:
        directory: Where the data lives; ``None`` reads ``FLY_CONNECTOME_DIR``.
        release: FlyWire release.
        num_neurons: Keep this many neurons and the synapses between them. ``None``
            keeps the whole brain.

    Raises:
        ValueError: If ``num_neurons`` is not positive.
        ConnectomeNotFoundError: If the data is missing.
    """
    if num_neurons is not None and num_neurons < 1:
        raise ValueError(f"num_neurons must be positive or None, got {num_neurons}")

    files = find_files(directory, release)
    completeness, connectivity = _read_tables(files)

    root_ids = np.asarray(completeness.index.to_numpy(), dtype=np.int64)
    pre = connectivity[_PRE_COLUMN].to_numpy(dtype=np.int64)
    post = connectivity[_POST_COLUMN].to_numpy(dtype=np.int64)
    weight = connectivity[_WEIGHT_COLUMN].to_numpy(dtype=np.float64)

    if num_neurons is not None and num_neurons < len(root_ids):
        root_ids = root_ids[:num_neurons]
        keep = (pre < num_neurons) & (post < num_neurons)
        pre, post, weight = pre[keep], post[keep], weight[keep]

    return Subnetwork(root_ids=root_ids, pre=pre, post=post, weight=weight, release=files.release)


def subnetwork_from_arrays(
    root_ids: Any, pre: Any, post: Any, weight: Any, release: str = "synthetic"
) -> Subnetwork:
    """Build a :class:`Subnetwork` from plain arrays.

    Exists so the backends can be tested against a hand-built network of a few
    neurons, with no 370 MB download and no simulator-specific fixtures.

    Raises:
        ValueError: If the three synapse arrays differ in length, or an index falls
            outside the neuron list.
    """
    root_id_array = np.asarray(root_ids, dtype=np.int64)
    pre_array = np.asarray(pre, dtype=np.int64)
    post_array = np.asarray(post, dtype=np.int64)
    weight_array = np.asarray(weight, dtype=np.float64)
    lengths = {len(pre_array), len(post_array), len(weight_array)}
    if len(lengths) != 1:
        raise ValueError(
            f"pre, post and weight must be the same length, got "
            f"{len(pre_array)}, {len(post_array)}, {len(weight_array)}"
        )
    limit = len(root_id_array)
    for name, array in (("pre", pre_array), ("post", post_array)):
        if len(array) and (array.min() < 0 or array.max() >= limit):
            raise ValueError(
                f"{name} indexes outside the {limit} neurons given "
                f"(min {array.min()}, max {array.max()})"
            )
    return Subnetwork(
        root_ids=root_id_array,
        pre=pre_array,
        post=post_array,
        weight=weight_array,
        release=release,
    )
