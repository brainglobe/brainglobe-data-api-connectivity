from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.io.loaders import load_test_data


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
