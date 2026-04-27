from pathlib import Path

import pandas as pd

from brainglobe_data_api_connectivity.connections import Connections


def test_me(
    DATA_DIR: Path,
    node_file: Path | str = "small-nodes.csv",
    edge_table: Path | str = "small-edge-table.csv",
    edge_meta: Path | str = "small-edge-meta.csv",
) -> None:
    """"""
    node_file = DATA_DIR / node_file
    edge_table = DATA_DIR / edge_table
    edge_meta = DATA_DIR / edge_meta

    G = Connections.from_files(node_file, edge_table, edge_meta)

    pass


def test_setup_network_nodes_reindexed(
    DATA_DIR,
    read_edge_table,
    nodes="small-nodes.csv",
    edge_table="small-edge-table.csv",
) -> None:
    """"""
    nodes = pd.read_csv(DATA_DIR / nodes, delimiter=",")
    edge_table = read_edge_table(DATA_DIR / edge_table)

    nodes_before = pd.DataFrame(nodes)

    G = Connections(edge_table, nodes)
    nodes_after = G.nodes

    # Don't drop any nodes
    assert nodes_after.shape == nodes_before.shape

    # Node mapping has taken place appropriately
    for old_index in nodes_before.index:
        new_index = nodes_after.index[old_index]

        old_row = nodes_before.iloc[old_index]
        new_row = nodes_after.iloc[new_index]

        pd.testing.assert_series_equal(
            old_row,
            new_row,
        )
