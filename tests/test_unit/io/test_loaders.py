from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.io.loaders import (
    load_data,
    load_test_data,
)


def test_load_test_data(DATA_DIR):
    """Downloaded connectivity data should match the local test data."""
    expected_data = Connections.from_files(
        node_info=DATA_DIR / "small-nodes.csv",
        edge_table=DATA_DIR / "small-edge-table.csv",
        edge_info=DATA_DIR / "small-edge-meta.csv",
    )
    loaded_test_data = load_test_data()

    assert_frame_equal(loaded_test_data.nodes, expected_data.nodes)
    assert_frame_equal(loaded_test_data.edge_info, expected_data.edge_info)
    assert loaded_test_data.network.nodes() == expected_data.network.nodes()
    assert (
        loaded_test_data.network.weighted_edge_list()
        == expected_data.network.weighted_edge_list()
    )


def test_load_data():
    """Download the real dataset and check its graph and metadata."""
    loaded_data = load_data()

    assert isinstance(loaded_data, Connections)
    assert loaded_data.nodes.height == 834
    assert loaded_data.network.num_nodes() == 834
    assert loaded_data.network.num_edges() == 74098
    assert loaded_data.network.get_edge_data(0, 1) == 6
    assert loaded_data.nodes["region_id"][0] == "GPl_1"
    assert loaded_data.edge_info.height == 48146
    assert loaded_data.edge_info_from_col == "origin_region_idx"
    assert loaded_data.edge_info_to_col == "termination_region_idx"
