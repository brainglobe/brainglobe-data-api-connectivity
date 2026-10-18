import polars as pl
import pytest
from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity.connections import (
    ConnectionLookupUnavailable,
    Connections,
)
from brainglobe_data_api_connectivity.connections.query_opts import (
    ConnectionsLookup,
    NodeIs,
)


@pytest.mark.parametrize(
    (
        "query_nodes",
        "node_as",
        "return_edges",
        "connections_lookup",
        "expected",
    ),
    [
        pytest.param(
            [0, 1],
            NodeIs.ANY,
            False,
            ConnectionsLookup.REPORTED,
            [
                {
                    "node_as": "input",
                    "name": "C",
                    "idx": 2,
                    "group": "C",
                    "notes": "C is part of group C",
                    "custom_index": 3,
                    "__node_index": 2,
                }
            ],
            id="Nodes (reported)",
        ),
        pytest.param(
            [{"name": "A"}, {"name": "B"}],
            NodeIs.ANY,
            True,
            ConnectionsLookup.REPORTED,
            [
                {"from": "A", "to": "C", "value": 10.0},
                {"from": "B", "to": "C", "value": 1.0},
            ],
            id="Edges using node dictionaries (reported)",
        ),
        pytest.param(
            [{"name": "A"}, {"name": "B"}],
            NodeIs.ANY,
            False,
            ConnectionsLookup.ALL,
            [
                {
                    "node_as": "input",
                    "name": "C",
                    "idx": 2,
                    "group": "C",
                    "notes": "C is part of group C",
                    "custom_index": 3,
                    "__node_index": 2,
                }
            ],
            id="Nodes (all connections)",
        ),
        pytest.param(
            [0, 1],
            NodeIs.INPUT,
            False,
            ConnectionsLookup.REPORTED,
            [
                {
                    "node_as": "input",
                    "name": "C",
                    "idx": 2,
                    "group": "C",
                    "notes": "C is part of group C",
                    "custom_index": 3,
                    "__node_index": 2,
                }
            ],
            id="Nodes (input direction)",
        ),
        pytest.param(
            [0, 3],
            NodeIs.OUTPUT,
            False,
            ConnectionsLookup.REPORTED,
            [],
            id="No nodes (output direction)",
        ),
        pytest.param(
            [0, 1],
            NodeIs.INPUT,
            True,
            ConnectionsLookup.REPORTED,
            [
                {"from": 0, "to": 2, "value": 10.0},
                {"from": 1, "to": 2, "value": 1.0},
            ],
            id="Edges (input direction)",
        ),
        pytest.param(
            [1, 2],
            NodeIs.OUTPUT,
            True,
            ConnectionsLookup.REPORTED,
            [
                {"from": 0, "to": 1, "value": 0.1},
                {"from": 0, "to": 2, "value": 10.0},
            ],
            id="Edges (output direction)",
        ),
        pytest.param(
            [0, 3],
            NodeIs.ANY,
            True,
            ConnectionsLookup.REPORTED,
            [
                {"from": 0, "to": 2, "value": 10.0},
                {"from": 2, "to": 3, "value": 10.0},
            ],
            id="Edges (mixed directions)",
        ),
        pytest.param(
            [0, 3],
            NodeIs.INPUT,
            True,
            ConnectionsLookup.REPORTED,
            [],
            id="No edges (input direction)",
        ),
        pytest.param(
            [1, 2],
            NodeIs.OUTPUT,
            False,
            ConnectionsLookup.REPORTED,
            [
                {
                    "node_as": "output",
                    "name": "A",
                    "idx": 0,
                    "group": "AB",
                    "notes": "A is part of group AB",
                    "custom_index": 1,
                    "__node_index": 0,
                }
            ],
            id="Nodes (output direction)",
        ),
        pytest.param(
            [0, 3],
            NodeIs.ANY,
            False,
            ConnectionsLookup.REPORTED,
            [
                {
                    "node_as": "mixed",
                    "name": "C",
                    "idx": 2,
                    "group": "C",
                    "notes": "C is part of group C",
                    "custom_index": 3,
                    "__node_index": 2,
                }
            ],
            id="Nodes (mixed directions)",
        ),
        pytest.param(
            [],
            NodeIs.ANY,
            False,
            ConnectionsLookup.REPORTED,
            [],
            id="Nodes (empty query)",
        ),
    ],
)
def test_common_connections_info(
    mini_G,
    query_nodes,
    node_as,
    return_edges,
    connections_lookup,
    expected,
) -> None:
    """Return node or edge information for common connections."""
    result = mini_G.common_connections_info(
        query_nodes,
        node_as=node_as,
        connections_lookup=connections_lookup,
        return_edges=return_edges,
    )
    assert_frame_equal(result, pl.DataFrame(expected, schema=result.schema))


def test_common_connections_info_no_edge_info(mini_G) -> None:
    """Raise when ALL is requested but edge metadata is unavailable."""
    mini_G.edge_info = None

    with pytest.raises(
        ConnectionLookupUnavailable,
        match="No edge information available",
    ):
        mini_G.common_connections_info(
            [0, 1],
            connections_lookup=ConnectionsLookup.ALL,
            return_edges=True,
        )


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
def test_common_connections_info_differs_from_network(
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


@pytest.mark.parametrize(
    ("node_as", "expected"),
    [
        pytest.param(
            NodeIs.INPUT,
            [
                {"idx": 2, "node_as": "input"},
            ],
            id="INPUT",
        ),
        pytest.param(
            NodeIs.OUTPUT,
            [
                {"idx": 4, "node_as": "output"},
            ],
            id="OUTPUT",
        ),
        pytest.param(
            NodeIs.ANY,
            [
                {"idx": 2, "node_as": "input"},
                {"idx": 3, "node_as": "mixed"},
                {"idx": 4, "node_as": "output"},
            ],
            id="ANY",
        ),
    ],
)
def test_common_connections_nodes_directions(node_as, expected) -> None:
    """Test common nodes classified by their connection direction.

    Representation of the network with nodes 0, 1, 2, 3, and 4:

       (0) ──────> (2) <───── (1)
        ^                       ^
        │                       │
        └──────── (4) ──────────┘

        (0) ──────> (3) ──────> (1)

    When querying common connections of nodes 0 and 1:
    INPUT:  Both queried nodes are inputs to node 2.
    OUTPUT: Both queried nodes are outputs of node 4.
    ANY:    Nodes 2, 3, and 4 are common connections.
            Node 2 is input, node 4 is output, and node 3 is mixed
            because node 0 is its input and node 1 is its output.

    """

    graph = Connections(
        node_info=pl.DataFrame({"idx": [0, 1, 2, 3, 4]}),
        edge_table=[
            (0, 2, 1.0),
            (1, 2, 1.0),
            (4, 0, 1.0),
            (4, 1, 1.0),
            (0, 3, 1.0),
            (3, 1, 1.0),
        ],
    )

    result = graph.common_connections_info(
        [0, 1],
        node_as=node_as,
        return_edges=False,
    )

    assert result.select("idx", "node_as").sort("idx").to_dicts() == expected
