import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.connections.query_opts import (
    ConnectionsLookup,
    NodeIs,
)


@pytest.fixture
def nodes() -> pl.DataFrame:
    """Simple collection of nodes to use when checking network setup."""
    return pl.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "idx": [0, 1, 2, 3],
            "group": ["AB", "AB", "C", "D"],
            "notes": [
                "A is part of group AB",
                "B is part of group AB",
                "C is part of group C",
                "D is part of group D",
            ],
            "custom_index": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def edge_list() -> list[tuple[int, int, float]]:
    """Edge list fixture compatible with Connections().

    Representation of the simple network with nodes A (0), B (1), C (2), and
    D (3):

    (A) ── 0.1 ──▶ (B) ── 10.0 ──▶ (C) ── 10.0 ──▶ (D)
     │                               ▲
     │                               │
     └───────────── 1.0 ─────────────┘

    """
    return [
        (0, 1, 0.1),
        (1, 2, 1.0),
        (0, 2, 10),
        (2, 3, 10),
    ]


@pytest.fixture
def edge_info() -> pl.DataFrame:
    "A simple 'edge information' frame for testing setup."
    return pl.DataFrame(
        {
            "from_id": ["A", "B", "A", "A", "C"],
            "to_id": ["B", "C", "C", "C", "D"],
            "from": [0, 1, 0, 0, 2],
            "to": [1, 2, 2, 2, 3],
            "used": ["yes", "yes", "yes", "no", "yes"],
            "paper": [
                "author et al., 1998",
                "author et al., 2025",
                "author et al., 2010",
                "author et al., 2020",
                "author et al., 2020",
            ],
            "strength": [
                "weak (0.1)",
                "medium (1.0)",
                "strong (10.0)",
                "medium (1.0)",
                "strong (10.0)",
            ],
        }
    )


@pytest.fixture
def mini_G(nodes, edge_list, edge_info) -> Connections:
    """Small Connections instance for testing"""
    return Connections(nodes, edge_list, edge_info)


@pytest.mark.parametrize(
    [
        "node0",
        "node1",
        "node0_as",
        "connections_lookup",
        "expected_bool",
        "expected_shape",
    ],
    [
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            True,
            (1, 9),
            id="A and B",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.ANY,
            ConnectionsLookup.REPORTED,
            True,
            (1, 3),
            id="B and A",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            True,
            (2, 9),
            id="A and C (all)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            NodeIs.ANY,
            ConnectionsLookup.REPORTED,
            True,
            (1, 3),
            id="A and C (reported only)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            NodeIs.INPUT,
            ConnectionsLookup.REPORTED,
            True,
            (1, 3),
            id="A (as INPUT) and C (reported only)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            NodeIs.OUTPUT,
            ConnectionsLookup.REPORTED,
            False,
            (0, 0),
            id="A (as OUTPUT) and C (reported only)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "D"},
            NodeIs.ANY,
            ConnectionsLookup.REPORTED,
            False,
            (0, 0),
            id="A and D (reported only)",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            True,
            (1, 9),
            id="B to A",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.INPUT,
            ConnectionsLookup.ALL,
            False,
            (0, 9),
            id="B (as INPUT) and A (no direct connection)",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.OUTPUT,
            ConnectionsLookup.REPORTED,
            True,
            (1, 3),
            id="B (as OUTPUT) and A (reported only)",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.OUTPUT,
            ConnectionsLookup.ALL,
            True,
            (1, 9),
            id="B (as OUTPUT) and A",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            NodeIs.INPUT,
            ConnectionsLookup.ALL,
            True,
            (1, 9),
            id="A (as INPUT) and B",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            NodeIs.OUTPUT,
            ConnectionsLookup.ALL,
            False,
            (0, 9),
            id="A (as OUTPUT) and B (no direct connection)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "D"},
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            False,
            (0, 9),
            id="A and D (no direct connection)",
        ),
    ],
)
def test_has_direct_connection_between(
    mini_G,
    node0,
    node1,
    node0_as,
    connections_lookup,
    expected_bool,
    expected_shape,
):
    """Test finding direct connections between two nodes."""
    has_connection, connections = mini_G.direct_connection_between(
        node0,
        node1,
        node0_as=node0_as,
        connections_lookup=connections_lookup,
    )

    assert has_connection == expected_bool
    assert connections.shape == expected_shape


def test_direct_connection_between_no_edge_info(mini_G):
    """Test graph fallback when edge information is unavailable."""
    mini_G.edge_info = None

    with pytest.warns(
        UserWarning,
        match="No edge information available. "
        "Using graph information instead.",
    ):
        has_connection, connections = mini_G.direct_connection_between(
            {"name": "A"},
            {"name": "C"},
            node0_as=NodeIs.ANY,
            connections_lookup=ConnectionsLookup.ALL,
        )

    assert has_connection is True
    assert connections.shape == (1, 3)


def test_get_unique_node_index_nonexistent_node(mini_G):
    """Test error when a node does not exist."""
    with pytest.raises(
        ValueError,
        match="Expected 1 unique node, but got 0",
    ):
        mini_G._get_unique_node_index({"name": "NONEXISTENT"})


def test_get_unique_node_index_non_unique_node(nodes, edge_list, edge_info):
    """Test error when a node is not unique."""
    nodes = nodes.with_columns(pl.col("name").replace("C", "A"))
    mini_G = Connections(nodes, edge_list, edge_info)

    assert len(mini_G.node_indexes_from_information(name="A")) == 2

    with pytest.raises(
        ValueError,
        match="Expected 1 unique node, but got 2",
    ):
        mini_G._get_unique_node_index({"name": "A"})
