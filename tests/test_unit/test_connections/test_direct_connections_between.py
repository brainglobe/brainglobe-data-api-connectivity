import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.connections.query_opts import (
    ConnectionsLookup,
    NodeIs,
)


@pytest.mark.parametrize(
    [
        "node0",
        "node1",
        "node0_as",
        "connections_lookup",
        "filters",
        "columns",
        "expected_shape",
    ],
    [
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            None,
            None,
            (1, 9),
            id="A and B",
        ),
        pytest.param(
            0,
            1,
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            None,
            None,
            (1, 9),
            id="A and B (by index)",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.ANY,
            ConnectionsLookup.NETWORK,
            None,
            None,
            (1, 3),
            id="B and A (reported only)",
        ),
        pytest.param(
            1,
            0,
            NodeIs.ANY,
            ConnectionsLookup.NETWORK,
            None,
            None,
            (1, 3),
            id="B and A by index (reported only)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            None,
            None,
            (2, 9),
            id="A and C (all)",
        ),
        pytest.param(
            0,
            2,
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            {"used": "yes"},
            None,
            (1, 9),
            id="A and C (all) + filter (used connections)",
        ),
        pytest.param(
            0,
            2,
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            {"used": "no"},
            None,
            (1, 9),
            id="A and C (all) + filter (unused connections)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            NodeIs.ANY,
            ConnectionsLookup.NETWORK,
            None,
            None,
            (1, 3),
            id="A and C (reported only)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            NodeIs.INPUT,
            ConnectionsLookup.NETWORK,
            None,
            None,
            (1, 3),
            id="A (as INPUT) and C (reported only)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            NodeIs.OUTPUT,
            ConnectionsLookup.NETWORK,
            None,
            None,
            (0, 0),
            id="A (as OUTPUT) and C (reported only)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "D"},
            NodeIs.ANY,
            ConnectionsLookup.NETWORK,
            None,
            None,
            (0, 0),
            id="A and D (reported only)",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            None,
            None,
            (1, 9),
            id="B to A",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.INPUT,
            ConnectionsLookup.ALL,
            None,
            None,
            (0, 9),
            id="B (as INPUT) and A (no direct connection)",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.OUTPUT,
            ConnectionsLookup.NETWORK,
            None,
            None,
            (1, 3),
            id="B (as OUTPUT) and A (reported only)",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.OUTPUT,
            ConnectionsLookup.ALL,
            None,
            None,
            (1, 9),
            id="B (as OUTPUT) and A",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            NodeIs.INPUT,
            ConnectionsLookup.ALL,
            None,
            None,
            (1, 9),
            id="A (as INPUT) and B",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            NodeIs.OUTPUT,
            ConnectionsLookup.ALL,
            None,
            None,
            (0, 9),
            id="A (as OUTPUT) and B (no direct connection)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "D"},
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            None,
            None,
            (0, 9),
            id="A and D (no direct connection)",
        ),
        pytest.param(
            0,
            2,
            NodeIs.INPUT,
            ConnectionsLookup.ALL,
            {"used": "no", "strength": "medium (1.0)"},
            None,
            (1, 9),
            id="Multiple filters",
        ),
        pytest.param(
            0,
            2,
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            None,
            ["strength", "paper"],
            (2, 2),
            id="Multiple columns",
        ),
        pytest.param(
            0,
            2,
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            {"used": "no"},
            ["paper"],
            (1, 1),
            id="filter and one column",
        ),
        pytest.param(
            0,
            2,
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            {"used": "yes", "strength": "medium (1.0)"},
            ["paper", "strength"],
            (0, 2),
            id="A and C (no matching filtered connections)",
        ),
    ],
)
def test_direct_connection_between(
    mini_G,
    node0,
    node1,
    node0_as,
    connections_lookup,
    filters,
    columns,
    expected_shape,
):
    """Test finding direct connections between two nodes."""
    connections = mini_G.direct_connection_between(
        node0,
        node1,
        node0_as=node0_as,
        connections_lookup=connections_lookup,
        filters=filters,
        columns=columns,
    )

    assert connections.shape == expected_shape


@pytest.mark.parametrize(
    ["options", "message"],
    [
        pytest.param(
            {"filters": {"used": "INVALID_VALUE"}},
            "Unknown filter value",
            id="Invalid filter value",
        ),
        pytest.param(
            {"columns": ["INVALID_COLUMN"]},
            "Unknown columns",
            id="Invalid column",
        ),
        pytest.param(
            {"columns": [0]},
            "Unknown columns",
            id="Invalid column (int index)",
        ),
    ],
)
def test_direct_connection_between_filter_column_warnings(
    mini_G, options, message
):
    """Warn for ignored options and reject invalid metadata options."""
    with pytest.raises(ValueError, match=message):
        mini_G.direct_connection_between(
            0, 2, connections_lookup=ConnectionsLookup.ALL, **options
        )


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(
            {"filters": {"used": "yes"}}, id="Valid filter (NETWORK)"
        ),
        pytest.param(
            {
                "filters": {"used": "no", "strength": "medium (1.0)"},
                "columns": ["paper", "strength"],
            },
            id="Valid multiple filters and columns (NETWORK)",
        ),
        pytest.param({"columns": ["paper"]}, id="Valid column (NETWORK)"),
    ],
)
def test_direct_connection_filters_ignored(mini_G, options):
    """Warn when valid filters or columns are ignored for network lookup."""
    with pytest.warns(UserWarning, match="filters and columns are ignored"):
        mini_G.direct_connection_between(0, 2, **options)


def test_direct_connection_between_no_edge_info(mini_G):
    """Test graph fallback when edge information is unavailable."""
    mini_G.edge_info = None

    with pytest.warns(
        UserWarning,
        match="No edge information available. "
        "Using graph information instead.",
    ):
        connections = mini_G.direct_connection_between(
            {"name": "A"},
            {"name": "C"},
            node0_as=NodeIs.ANY,
            connections_lookup=ConnectionsLookup.ALL,
        )

    assert connections.shape == (1, 3)


def test_direct_connection_between_nonexistent_node(mini_G):
    """Test error when a node does not exist."""

    assert len(mini_G.node_indexes_from_information(name="NONEXISTENT")) == 0

    with pytest.raises(
        ValueError,
        match="Expected 1 unique node, but got 0",
    ):
        mini_G.direct_connection_between(
            {"name": "NONEXISTENT"},
            {"name": "A"},
        )


def test_direct_connection_between_non_unique_node(
    nodes,
    edge_list,
    edge_info,
):
    """Test error when a node is not unique."""
    nodes = nodes.with_columns(pl.col("name").replace("C", "A"))
    mini_G = Connections(nodes, edge_list, edge_info)

    assert len(mini_G.node_indexes_from_information(name="A")) == 2

    with pytest.raises(
        ValueError,
        match="Expected 1 unique node, but got 2",
    ):
        mini_G.direct_connection_between(
            {"name": "A"},
            {"name": "B"},
        )
