import pandas as pd

from brainglobe_data_api_connectivity.connections import Connections


def test_setup_network_nodes_reindexed(
    DATA_DIR,
    read_edge_table,
    nodes="small-nodes.csv",
    edge_table="small-edge-table.csv",
) -> None:
    """Tests that, if nodes are provided using a different convention to row
    index identifiers, re-indexing via the network setup is handled correctly.
    """
    nodes = pd.read_csv(DATA_DIR / nodes, delimiter=",")
    # Introduce a non-standard indexing of the input nodes,
    # reflecting that we are not using the row-number as the
    # index.
    nodes.index = reversed(nodes.index)

    edge_table = read_edge_table(DATA_DIR / edge_table)
    nodes_before = pd.DataFrame(nodes)

    G = Connections(edge_table, nodes)
    nodes_after = G.nodes

    # Don't drop any nodes
    assert nodes_after.shape == nodes_before.shape

    # Node mapping has taken place appropriately.
    # Indexes may have swapped, but information has been kept
    # consistent
    for i, old_index in enumerate(nodes_before.index):
        new_index = nodes_after.index[i]

        old_row = nodes_before.loc[old_index]
        new_row = nodes_after.loc[new_index]

        pd.testing.assert_series_equal(old_row, new_row, check_names=False)
        assert old_row.name == old_index
        assert new_row.name == new_index

    # Edges should also map consistently
    assert len(edge_table) == len(G.network.edge_list())
    for old_from, old_to, old_weight in edge_table:
        new_from = nodes_after.index[old_from]
        new_to = nodes_after.index[old_to]
        new_weight = G.network.get_edge_data(new_from, new_to)

        assert old_weight == new_weight
