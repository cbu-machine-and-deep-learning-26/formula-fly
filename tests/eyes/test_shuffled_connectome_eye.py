"""Degree-matched shuffled connectome eye (GH-15).

The shuffle itself is tested without flyvis in ``tests/test_degree_matched_shuffle.py``.
These tests cover the flyvis wiring: a missing install, the tensors the
dynamics actually read, and (when a checkpoint is present) the real optic lobe.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from fly_driver.eyes.constants import FLYVIS_DEFAULT_FEATURE_DIM
from fly_driver.eyes.flyvis_eye import (
    DEFAULT_MOTION_READOUTS,
    FlyvisNotInstalledError,
    resolve_checkpoint_dir,
)
from fly_driver.eyes.hex_resampler import HEX_COLUMN_COUNT, RGB_LUMA_WEIGHTS
from fly_driver.eyes.random_projection_eye import _LUMA_WEIGHTS
from fly_driver.eyes.shuffled_connectome_eye import ShuffledConnectomeEye, rewire_network

torch = pytest.importorskip("torch")


def test_feature_width_is_the_t4_t5_readout() -> None:
    assert len(DEFAULT_MOTION_READOUTS) * HEX_COLUMN_COUNT == FLYVIS_DEFAULT_FEATURE_DIM
    np.testing.assert_allclose(_LUMA_WEIGHTS, RGB_LUMA_WEIGHTS)


def test_missing_flyvis_raises_the_same_error_as_the_real_eye(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "flyvis", None)
    with pytest.raises(FlyvisNotInstalledError, match="requirements-flyvis.txt"):
        ShuffledConnectomeEye()


def test_a_bad_seed_fails_before_flyvis_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "flyvis", None)
    with pytest.raises(ValueError, match="seed"):
        ShuffledConnectomeEye(seed=-1)
    with pytest.raises(TypeError, match="seed"):
        ShuffledConnectomeEye(seed=True)  # type: ignore[arg-type]


def test_rewire_updates_the_tensors_and_the_cached_readers() -> None:
    """flyvis gathers parameters through readers captured at init, not only the index tensors."""
    sources = np.array([0, 1, 2], dtype=np.int64)
    targets = np.array([1, 2, 0], dtype=np.int64)
    new_targets = np.array([2, 0, 1], dtype=np.int64)

    class Param:
        def __init__(self) -> None:
            self.indices = torch.tensor([0, 1, 2])
            self.readers = {
                "nodes": self.indices,
                "sources": self.indices[torch.as_tensor(sources)],
                "targets": self.indices[torch.as_tensor(targets)],
            }

    class Edges:
        def __init__(self) -> None:
            self.source_index = sources.copy()
            self.target_index = targets.copy()

    class Connectome:
        def __init__(self) -> None:
            self.edges = Edges()

    class Network:
        def __init__(self) -> None:
            self._source_indices = torch.as_tensor(sources)
            self._target_indices = torch.as_tensor(targets)
            self.node_params = {"time_const": Param()}
            self.connectome = Connectome()

    network = Network()
    rewire_network(network, sources, new_targets)
    assert torch.equal(network._target_indices, torch.as_tensor(new_targets))
    param = network.node_params["time_const"]
    assert torch.equal(param.readers["targets"], param.indices[network._target_indices])
    assert torch.equal(param.readers["sources"], param.indices[network._source_indices])
    np.testing.assert_array_equal(network.connectome.edges.target_index, new_targets)


def test_rewire_rejects_a_different_edge_count() -> None:
    class Network:
        def __init__(self) -> None:
            self._source_indices = torch.zeros(2, dtype=torch.long)
            self._target_indices = torch.ones(2, dtype=torch.long)

    with pytest.raises(ValueError, match="edges"):
        rewire_network(Network(), np.array([0]), np.array([1]))


def test_pretrained_shuffle_preserves_degree_and_readout_width() -> None:
    """The optic-lobe edge list, not a stand-in, keeps its degrees."""
    pytest.importorskip("flyvis")
    try:
        resolve_checkpoint_dir()
    except FileNotFoundError:
        pytest.skip("run `flyvis download-pretrained` to enable this test")
    except FlyvisNotInstalledError:
        pytest.skip("flyvis is not installed")
    eye = ShuffledConnectomeEye(seed=0, swap_attempts=2_000)
    assert isinstance(eye.feature_dim, int)
    assert eye.feature_dim == FLYVIS_DEFAULT_FEATURE_DIM
    sources = eye.network._source_indices.detach().cpu().numpy()
    targets = eye.network._target_indices.detach().cpu().numpy()
    from fly_driver.eyes.degree_shuffle import in_out_degrees

    after_in, after_out = in_out_degrees(sources, targets, int(eye.network.n_nodes))
    np.testing.assert_array_equal(after_in, eye.in_degree)
    np.testing.assert_array_equal(after_out, eye.out_degree)
    assert not np.any(sources == targets)
