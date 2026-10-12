"""Load connectivity data from GIN."""

import pooch

from ..connections import Connections

GIN_URL = (
    "https://gin.swc.ucl.ac.uk/brainglobe/data-api-connectivity/raw/master/"
)


def load_test_data() -> Connections:
    """Load the test connectivity dataset from GIN."""
    path_node_info = pooch.retrieve(
        url=f"{GIN_URL}test-data/small-nodes.csv",
    )

    path_edge_table = pooch.retrieve(
        url=f"{GIN_URL}test-data/small-edge-table.csv",
    )

    path_edge_info = pooch.retrieve(
        url=f"{GIN_URL}test-data/small-edge-meta.csv",
    )

    return Connections.from_files(
        node_info=path_node_info,
        edge_table=path_edge_table,
        edge_info=path_edge_info,
    )
