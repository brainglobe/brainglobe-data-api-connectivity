import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.connections.query_opts import (
    ConnectionsLookup,
    NodeIs,
)


@pytest.mark.parametrize(
    ("query_nodes", "node_as", "expected"),
    [
        pytest.param(
            [0, 1],
            NodeIs.ANY,
            [
                {
                    "name": "C",
                    "idx": 2,
                    "group": "C",
                    "notes": "C is part of group C",
                    "custom_index": 3,
                    "__node_index": 2,
                }
            ],
            id="common connection",
        ),
        pytest.param(
            [0, 1],
            NodeIs.INPUT,
            [
                {
                    "name": "C",
                    "idx": 2,
                    "group": "C",
                    "notes": "C is part of group C",
                    "custom_index": 3,
                    "__node_index": 2,
                }
            ],
            id="common output",
        ),
        pytest.param(
            [0, 3],
            NodeIs.OUTPUT,
            [],
            id="no common connections",
        ),
    ],
)
def test_common_connections_info(
    mini_G, query_nodes, node_as, expected
) -> None:
    """Return node information for common connections."""
    result = mini_G.common_connections_node_info(query_nodes, node_as=node_as)
    assert result.to_dicts() == expected


@pytest.mark.parametrize(
    ("node_indices", "node_as", "expected"),
    [
        pytest.param(
            [0, 1],
            NodeIs.ANY,
            [2],
            id="common connections in either direction",
        ),
        pytest.param(
            [0, 1], NodeIs.INPUT, [2], id="common output of nodes 0 and 1"
        ),
        pytest.param(
            [1, 2], NodeIs.OUTPUT, [0], id="common input to nodes 1 and 2"
        ),
    ],
)
@pytest.mark.parametrize(
    "connections_lookup",
    [
        pytest.param(ConnectionsLookup.ALL, id="all edge information"),
        pytest.param(ConnectionsLookup.REPORTED, id="in network"),
    ],
)
def test_common_connections(
    mini_G,
    node_indices,
    node_as,
    expected,
    connections_lookup,
) -> None:
    """Return nodes directly connected to all given nodes."""
    common = mini_G.common_connections(
        node_indices,
        node_as=node_as,
        connections_lookup=connections_lookup,
    )
    assert common == expected


@pytest.mark.parametrize(
    ("edge_table", "node_as"),
    [
        pytest.param(
            [(1, 3, 1.0), (2, 3, 1.0)],
            NodeIs.INPUT,
            id="nodes 1 and 2 are inputs to 3",
        ),
        pytest.param(
            [(3, 1, 1.0), (3, 2, 1.0)],
            NodeIs.OUTPUT,
            id="nodes 1 and 2 are outputs of 3",
        ),
    ],
)
def test_common_connections_direction_examples(edge_table, node_as) -> None:
    """Nodes 1 and 2 share node 3 in the specified direction only."""
    graph = Connections(
        node_info=pl.DataFrame({"idx": [0, 1, 2, 3]}),
        edge_table=edge_table,
    )

    assert graph.common_connections([1, 2], node_as=node_as) == [3]


@pytest.mark.parametrize(
    ("connections_lookup", "expected"),
    [
        (ConnectionsLookup.ALL, [2, 3]),
        (ConnectionsLookup.REPORTED, [2]),
    ],
)
def test_common_connections_edge_info_differs_from_network(
    mini_G,
    connections_lookup,
    expected,
) -> None:
    """Test common connections when edge info and network differ.

    In this test A and B both connect to D in `edge_info`, but these
    connections are not present in the network. So D [3] is only a
    common connection when querying all edge information."""
    extra_edges = pl.DataFrame(
        {
            "from_id": ["A", "B"],
            "to_id": ["D", "D"],
            "from": [0, 1],
            "to": [3, 3],
            "used": ["no", "no"],
            "paper": ["", ""],
            "strength": ["medium (1.0)", "medium (1.0)"],
            "__idx_from": [0, 1],
            "__idx_to": [3, 3],
        }
    )

    mini_G.edge_info = pl.concat([mini_G.edge_info, extra_edges])

    assert (
        mini_G.common_connections(
            [0, 1],
            node_as=NodeIs.INPUT,
            connections_lookup=connections_lookup,
        )
        == expected
    )
