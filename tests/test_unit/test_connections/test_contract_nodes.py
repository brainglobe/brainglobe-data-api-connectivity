from pathlib import Path

import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections


@pytest.mark.parametrize(
    ("node_info", "edge_table", "nodes_to_contract"),
    [
        pytest.param(
            "small-nodes.csv",
            "small-edge-table.csv",
            (1,),
            id="Single node contract",
        ),
        pytest.param(
            "small-nodes.csv",
            "small-edge-table.csv",
            (0, 1),
            id="2-node contract",
        ),
        pytest.param(
            "small-nodes.csv",
            "small-edge-table.csv",
            (0, 1, 2, 3, 4),
            id="Collapse all the nodes",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "name": ["a", "a", "b", "c", "d"],
                }
            ),
            "small-edge-table.csv",
            (0, 1),
            id="Collapse with identical nodes (included in contract)",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "name": ["a", "a", "b", "c", "d"],
                }
            ),
            "small-edge-table.csv",
            (0, 2),
            id="Collapse with identical nodes (excluded from contract)",
        ),
    ],
)
def test_contract_nodes(
    node_info: str | Path,
    edge_table: str | Path,
    nodes_to_contract: tuple[int],
    DATA_DIR: Path,
    tmp_csv,
) -> None:
    """Confirm that contracting nodes acts as intended.

    Note that the actual collapsing is handled by `rustworkx` under the hood,
    and we directly call their method. As such, we do not check in details that
    the contract has been carried out on the network itself correctly, though
    we do make a sanity check that the contracted nodes are indeed gone.
    Similarly, we do not check that the weight function has been applied
    correctly, since this is directly delegated to the `rustworkx` method.
    """
    node_info = (
        DATA_DIR / node_info
        if isinstance(node_info, str | Path)
        else tmp_csv(node_info, "nodes.csv")
    )
    edge_table = (
        DATA_DIR / edge_table
        if isinstance(edge_table, str | Path)
        else tmp_csv(edge_table, "edges.csv")
    )

    G = Connections.from_files(DATA_DIR / node_info, DATA_DIR / edge_table)

    # List of dicts, each dict being a row of the .nodes property that will
    # have it's internal index set to Null
    nodes_to_be_contracted = G.node_information_from_index(
        nodes_to_contract
    ).to_dicts()

    super_index = G.contract_nodes(nodes_to_contract)
    network_node_indexes_after_contract = G.network.node_indices()
    assert super_index in G.network.node_indices()
    assert all(
        i not in network_node_indexes_after_contract for i in nodes_to_contract
    )

    for row in nodes_to_be_contracted:
        # This is the node information, excluding the internal index.
        row_without_index = dict(row)
        lost_index = row_without_index.pop(G._node_internal_index_col)

        # exprs will find all nodes with the same information as the one we are
        # examining, ignoring the internal index. Duplicates are thus possible.
        exprs = [
            pl.col(column_name) == value
            for column_name, value in row_without_index.items()
        ]
        exprs.append(pl.col(G._node_internal_index_col).is_null())

        # node_index_after is not guaranteed to be of length 1, since nodes
        # with the same information up to the internal index are permitted,
        # and we have just dropped the internal index. As such, if 2+ nodes
        # with the same information are both part of the same contract,
        # fetching the node indexes will return more than one entry.
        n_nodes_that_should_have_contracted = sum(
            1
            for r in nodes_to_be_contracted
            if {
                col: val
                for col, val in r.items()
                if col != G._node_internal_index_col
            }
            == row_without_index
        )
        # At least one node - that is the one we are looking at - should have
        # been contracted!
        assert n_nodes_that_should_have_contracted >= 1

        node_index_after = G.node_indexes_from_information(*exprs)
        assert (
            node_index_after.null_count()
            == n_nodes_that_should_have_contracted
        )
        # Confirm that any remaining indexes of matching nodes are not equal to
        # the internal node index that we deleted.
        assert (node_index_after.drop_nulls() != lost_index).all()
