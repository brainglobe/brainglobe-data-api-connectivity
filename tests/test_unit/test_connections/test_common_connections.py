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
            True,
            ConnectionsLookup.ALL,
            [
                {
                    "from_id": "B",
                    "to_id": "C",
                    "from": 1,
                    "to": 2,
                    "used": "yes",
                    "paper": "author et al., 2025",
                    "strength": "medium (1.0)",
                    "__idx_from": 1,
                    "__idx_to": 2,
                },
                {
                    "from_id": "A",
                    "to_id": "C",
                    "from": 0,
                    "to": 2,
                    "used": "yes",
                    "paper": "author et al., 2010",
                    "strength": "strong (10.0)",
                    "__idx_from": 0,
                    "__idx_to": 2,
                },
                {
                    "from_id": "A",
                    "to_id": "C",
                    "from": 0,
                    "to": 2,
                    "used": "no",
                    "paper": "author et al., 2020",
                    "strength": "medium (1.0)",
                    "__idx_from": 0,
                    "__idx_to": 2,
                },
            ],
            id="Edges using node dictionaries (all connections)",
        ),
        pytest.param(
            [{"name": "A"}, {"name": "B"}],
            NodeIs.OUTPUT,
            True,
            ConnectionsLookup.ALL,
            [],
            id="Edges (common output) (all connections)",
        ),
        pytest.param(
            [{"name": "A"}, {"name": "C"}],
            NodeIs.INPUT,
            True,
            ConnectionsLookup.ALL,
            [],
            id="Edges (common input) (all connections)",
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
            id="Nodes using node dictionaries (all connections)",
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
    ("connections_lookup", "expected"),
    [
        pytest.param(ConnectionsLookup.ALL, [2, 3], id="ALL"),
        pytest.param(ConnectionsLookup.REPORTED, [2], id="REPORTED"),
    ],
)
def test_common_connections_info_differs_from_network(
    mini_G,
    connections_lookup,
    expected,
) -> None:
    """Test common nodes when edge information and network differ.

    Verify that ConnectionsLookup.ALL can return common nodes that
    ConnectionsLookup.REPORTED cannot, because those connections exist only in
    edge_info, not in the graph.

    Nodes 0 and 1 both connect to node 3 in edge_info, but these
    connections are absent from the network. Node 3 is therefore
    only common when querying ALL connections.
    """
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

    result = mini_G.common_connections_info(
        [0, 1],
        node_as=NodeIs.INPUT,
        connections_lookup=connections_lookup,
        return_edges=False,
    )

    assert result["idx"].sort().to_list() == expected


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

    assert_frame_equal(
        result.select("idx", "node_as").sort("idx"),
        pl.DataFrame(expected),
    )
