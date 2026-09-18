"""Connectome loading and subsetting (GH-23).

None of this needs a simulator or the 370 MB download: the loader is separated from
the backends precisely so the awkward parts -- finding the data, carving a subset,
rejecting a malformed one -- can be tested from a handful of synthetic neurons.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.brains import connectome


class TestFindingTheData:
    def test_it_says_which_variable_to_set_when_there_is_no_hint(self, monkeypatch):
        """The data is a deliberate download. The error has to say so, or the next
        person assumes the repository is broken."""
        monkeypatch.delenv(connectome.CONNECTOME_DIR_ENV, raising=False)
        with pytest.raises(connectome.ConnectomeNotFoundError, match="FLY_CONNECTOME_DIR"):
            connectome.resolve_connectome_dir()

    def test_the_environment_variable_is_used(self, monkeypatch, tmp_path):
        monkeypatch.setenv(connectome.CONNECTOME_DIR_ENV, str(tmp_path))
        assert connectome.resolve_connectome_dir() == tmp_path

    def test_an_explicit_path_beats_the_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv(connectome.CONNECTOME_DIR_ENV, str(tmp_path / "nope"))
        assert connectome.resolve_connectome_dir(tmp_path) == tmp_path

    def test_a_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(connectome.ConnectomeNotFoundError):
            connectome.resolve_connectome_dir(tmp_path / "absent")

    def test_an_unknown_release_lists_the_known_ones(self, tmp_path):
        with pytest.raises(ValueError, match="630"):
            connectome.find_files(tmp_path, release="999")

    def test_a_half_present_release_names_the_missing_file(self, tmp_path):
        """One file downloaded and not the other is the likely failure, and the
        message should not just say 'not found'."""
        (tmp_path / "Completeness_783.csv").write_text("root_id,Completed\n")
        with pytest.raises(connectome.ConnectomeNotFoundError, match="Connectivity_783"):
            connectome.find_files(tmp_path, release="783")


class TestBuildingFromArrays:
    def test_it_counts_what_it_was_given(self):
        network = connectome.subnetwork_from_arrays(
            root_ids=[10, 11, 12], pre=[0, 1], post=[1, 2], weight=[3.0, -2.0]
        )
        assert network.num_neurons == 3
        assert network.num_synapses == 2
        assert "3" in repr(network) and "synthetic" in repr(network)

    def test_inhibition_survives_as_a_negative_weight(self):
        """The connectome's weight column is a *signed* synapse count. Taking an
        absolute value somewhere would turn every inhibitory synapse excitatory and
        the network would look plausible and behave nothing like a fly."""
        network = connectome.subnetwork_from_arrays([0, 1], [0], [1], [-5.0])
        assert network.weight[0] == -5.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pre": [0, 1], "post": [1], "weight": [1.0, 1.0]},
            {"pre": [0], "post": [1], "weight": [1.0, 2.0]},
        ],
    )
    def test_ragged_synapse_arrays_are_rejected(self, kwargs):
        with pytest.raises(ValueError, match="same length"):
            connectome.subnetwork_from_arrays(root_ids=[0, 1], **kwargs)

    @pytest.mark.parametrize("pre, post", [([0, 5], [1, 1]), ([0, 0], [1, -1])])
    def test_indices_outside_the_neuron_list_are_rejected(self, pre, post):
        """An out-of-range index is a silent corruption in a sparse matrix, not a
        crash: it either drops the synapse or wraps it onto the wrong neuron."""
        with pytest.raises(ValueError, match="indexes outside"):
            connectome.subnetwork_from_arrays([0, 1], pre, post, [1.0, 1.0])

    def test_an_empty_synapse_list_is_allowed(self):
        """A network of disconnected neurons is a legitimate control condition."""
        network = connectome.subnetwork_from_arrays([0, 1, 2], [], [], [])
        assert network.num_synapses == 0


class TestSubsetting:
    """`load_subnetwork` reads real files, so the subset logic is exercised through
    a tiny stand-in pair written to disk in the upstream format."""

    @pytest.fixture
    def tiny(self, tmp_path):
        pandas = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        root_ids = [100, 101, 102, 103]
        pandas.DataFrame({"Completed": [True] * 4}, index=root_ids).to_csv(
            tmp_path / "Completeness_783.csv"
        )
        pandas.DataFrame(
            {
                "Presynaptic_Index": [0, 1, 2, 0],
                "Postsynaptic_Index": [1, 2, 3, 3],
                "Excitatory x Connectivity": [1.0, -2.0, 3.0, 4.0],
            }
        ).to_parquet(tmp_path / "Connectivity_783.parquet")
        return tmp_path

    def test_it_loads_everything_by_default(self, tiny):
        network = connectome.load_subnetwork(tiny, release="783")
        assert network.num_neurons == 4
        assert network.num_synapses == 4
        assert network.release == "783"

    def test_a_subset_drops_synapses_that_leave_it(self, tiny):
        """Keeping a dangling synapse would point at a neuron that is no longer in
        the network -- which is the out-of-range corruption above, arrived at from
        the other direction."""
        network = connectome.load_subnetwork(tiny, release="783", num_neurons=2)
        assert network.num_neurons == 2
        assert network.num_synapses == 1  # only 0 -> 1 has both ends inside
        assert network.pre.max() < 2 and network.post.max() < 2

    def test_asking_for_more_neurons_than_exist_gives_all_of_them(self, tiny):
        network = connectome.load_subnetwork(tiny, release="783", num_neurons=999)
        assert network.num_neurons == 4

    def test_root_ids_come_back_in_index_order(self, tiny):
        """Backends address neurons by position; naming a spike afterwards means
        mapping back through this array, so the order has to be the file's."""
        network = connectome.load_subnetwork(tiny, release="783")
        assert list(network.root_ids) == [100, 101, 102, 103]

    def test_a_non_positive_subset_is_rejected(self, tiny):
        with pytest.raises(ValueError, match="positive"):
            connectome.load_subnetwork(tiny, release="783", num_neurons=0)

    def test_weights_keep_their_sign_through_the_loader(self, tiny):
        network = connectome.load_subnetwork(tiny, release="783")
        assert np.any(network.weight < 0)
