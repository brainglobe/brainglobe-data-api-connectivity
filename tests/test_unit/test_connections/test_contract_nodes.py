from pathlib import Path
from typing import Any

import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections


@pytest.mark.parametrize(
    (
        "node_info",
        "edge_table",
        "nodes_to_contract",
        "provide_super_node_info",
    ),
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

    Note that the actual contracting is handled by `rustworkx` under the hood,
    and we directly call their method. As such, we do not check in details that
    the contraction has been carried out on the network itself correctly,
    though we do make a sanity check that the contracted nodes are indeed gone.
    Similarly, we do not check that the weight function has been applied
    correctly, since this is directly delegated to the `rustworkx` method (and
    the correctness of the weight functions is tested elsewhere).
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
    assert not G.collapsed_node_indexes
    super_index = G.contract_nodes(nodes_to_contract)

    network_node_indexes_after_contract = G.network.node_indices()
    assert super_index in G.network.node_indices()
    assert super_index in G.nodes.get_column(G._node_internal_index_col)
    assert all(
        i not in network_node_indexes_after_contract for i in nodes_to_contract
    )
    assert set(nodes_to_contract).issubset(G.collapsed_node_indexes)


@pytest.mark.parametrize(
    ("super_node_info",),
    [
        pytest.param(
            {},
            id="Provide no information",
        ),
        pytest.param(
            {"name": "super node"},
            id="Provide single column information",
        ),
        pytest.param(
            {"not_present": 1.0},
            id="Ignore fields that are not columns",
        ),
        pytest.param(
            {
                "name": "super node",
                "custom_index": 10,
                "mass": 1e2,
                "reverse_custom_index": -10,
            },
            id="Provide complete information",
        ),
    ],
)
def test_contract_nodes_provide_information(
    super_node_info: dict[str, Any],
    DATA_DIR: Path,
    nodes_to_contract: tuple[int] = (0, 2),  # alpha and charlie
    node_info: str = "small-nodes.csv",
    edge_table: str = "small-edge-table.csv",
) -> None:
    """
    Test that the user can provide information about super-nodes when they
    are created as a result of a contraction.
    """
    G = Connections.from_files(DATA_DIR / node_info, DATA_DIR / edge_table)
    n_nodes_rows_pre_contract, n_nodes_cols = G.nodes.shape

    super_node_index = G.contract_nodes(
        nodes_to_contract, super_node_info=super_node_info
    )

    # This lookup should always succeed, otherwise the super-node was not added
    # to the `nodes` table.
    stored_super_node_info = G.node_information_from_index([super_node_index])
    assert len(G.nodes.columns) == n_nodes_cols, (
        "G.nodes shouldn't gain columns"
    )
    assert G.nodes.shape[0] == n_nodes_rows_pre_contract + 1, (
        "G.nodes should gain a row (from the super-node)"
    )
    assert stored_super_node_info.shape == (1, n_nodes_cols), (
        "Should be exactly one hit when looking up the super-node"
    )
    stored_super_node_info = stored_super_node_info.to_dicts()[0]

    # Check that the information was stored correctly.
    # Note that any information that we did not pass should be present in the
    # resulting table, but stored as `None` (or `pl.Null` if directly examining
    # the DataFrame entries).
    for col_name in G.nodes.columns:
        if col_name != G._node_internal_index_col:
            assert (
                super_node_info.get(col_name)
                == stored_super_node_info[col_name]
            )
        else:
            assert stored_super_node_info[col_name] == super_node_index
