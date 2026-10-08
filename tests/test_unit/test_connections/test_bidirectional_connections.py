import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.connections.query_opts import (
    ConnectionsLookup,
)


@pytest.fixture
def bidi_edge_list() -> list[tuple[int, int, float]]:
    """Edge list fixture compatible with Connections().

    Representation of the simple network with nodes A (0), B (1), C (2), and
    D (3):

    (A) <── 1 / 5 ──> (B) <── 10 / 2 ──> (C) ── 10 ──> (D)
     │                                    ^
     │                                    │
     └───────────────── 3 ────────────────┘

    """
    return [
        (0, 1, 5),
        (1, 0, 1),
        (1, 2, 2),
        (2, 1, 10),
        (0, 2, 3),
        (2, 3, 10),
    ]


@pytest.fixture
def bidi_edge_info() -> pl.DataFrame:
    """A simple 'edge information' frame for testing setup."""
    return pl.DataFrame(
        {
            "from_id": ["A", "B", "B", "C", "A", "C", "D", "C"],
            "to_id": ["B", "A", "C", "B", "C", "D", "C", "B"],
            "from": [0, 1, 1, 2, 0, 2, 3, 2],
            "to": [1, 0, 2, 1, 2, 3, 2, 1],
            "used": ["yes", "yes", "yes", "yes", "yes", "yes", "no", "no"],
            "strength": ["5", "1", "2", "10", "3", "10", "10", "1"],
        }
    )


@pytest.fixture
def mini_G_bidi(nodes, bidi_edge_list, bidi_edge_info) -> Connections:
    """Small Connections instance for testing.

    Bidirectional connection rows per node:

    REPORTED (graph):
        A: 2  (A <-> B)
        B: 4  (B <-> A, B <-> C)
        C: 2  (C <-> B)
        D: 0
        E: 0  (isolated)

    ALL (edge info):
        A: 2  (A <-> B)
        B: 5  (B <-> A, B <-> C; extra C -> B report)
        C: 5  (C <-> B, C <-> D; extra C -> B report)
        D: 2  (D <-> C)
        E: 0  (isolated)

    """
    nodes = nodes.vstack(
        pl.DataFrame(
            [("E", 4, "E", "E is isolated", 5)],
            schema=nodes.schema,
            orient="row",
        )
    )
    return Connections(nodes, bidi_edge_list, bidi_edge_info)


@pytest.mark.parametrize(
    [
        "node",
        "connections_lookup",
        "filters",
        "columns",
        "expected_shape",
    ],
    [
        pytest.param(
            {"name": "A"},
            ConnectionsLookup.ALL,
            None,
            None,
            (2, 8),
            id="A (all)",
        ),
        pytest.param(
            {"name": "A"},
            ConnectionsLookup.NETWORK,
            None,
            None,
            (2, 3),
            id="A (reported)",
        ),
        pytest.param(
            {"name": "B"},
            ConnectionsLookup.ALL,
            None,
            None,
            (5, 8),
            id="B (all)",
        ),
        pytest.param(
            1,
            ConnectionsLookup.ALL,
            None,
            None,
            (5, 8),
            id="B by index (all)",
        ),
        pytest.param(
            {"name": "B"},
            ConnectionsLookup.NETWORK,
            None,
            None,
            (4, 3),
            id="B (reported)",
        ),
        pytest.param(
            {"name": "D"},
            ConnectionsLookup.ALL,
            None,
            None,
            (2, 8),
            id="D (all)",
        ),
        pytest.param(
            {"name": "D"},
            ConnectionsLookup.NETWORK,
            None,
            None,
            (0, 0),
            id="D (reported)",
        ),
        pytest.param(
            1,
            ConnectionsLookup.ALL,
            {"used": "yes"},
            None,
            (4, 8),
            id="B filter (used)",
        ),
        pytest.param(
            1,
            ConnectionsLookup.ALL,
            {"used": "no"},
            None,
            (1, 8),
            id="B filter (unused)",
        ),
        pytest.param(
            1,
            ConnectionsLookup.ALL,
            None,
            ["from_id", "to_id"],
            (5, 2),
            id="B (two columns)",
        ),
        pytest.param(
            1,
            ConnectionsLookup.ALL,
            {"strength": "5"},
            ["from_id", "to_id"],
            (1, 2),
            id="B (filter and two columns)",
        ),
        pytest.param(
            1,
            ConnectionsLookup.ALL,
            {"used": "no", "strength": "2"},
            ["strength"],
            (0, 1),
            id="No matching filtered connections",
        ),
        pytest.param(
            {"name": "E"},
            ConnectionsLookup.ALL,
            {"used": "yes"},
            ["strength", "from_id"],
            (0, 2),
            id="No reciprocal pairs with filter and columns",
        ),
    ],
)
def test_bidirectional_connections(
    mini_G_bidi,
    node,
    connections_lookup,
    filters,
    columns,
    expected_shape,
):
    """Test finding all bidirectional connections for a node."""
    connections = mini_G_bidi.bidirectional_connections(
        node,
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
def test_bidirectional_connections_filter_column_errors(
    mini_G_bidi, options, message
):
    """Reject invalid filters or columns for edge information lookup."""
    with pytest.raises(ValueError, match=message):
        mini_G_bidi.bidirectional_connections(
            1, connections_lookup=ConnectionsLookup.ALL, **options
        )


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(
            {"filters": {"used": "yes"}}, id="Valid filter (NETWORK)"
        ),
        pytest.param(
            {
                "filters": {"used": "no", "strength": "1"},
                "columns": ["from_id", "to_id"],
            },
            id="Valid multiple filters and columns (NETWORK)",
        ),
        pytest.param({"columns": ["strength"]}, id="Valid column (NETWORK)"),
    ],
)
def test_bidirectional_connections_filters_ignored(mini_G_bidi, options):
    """Warn when valid filters or columns are ignored for network lookup."""
    with pytest.warns(UserWarning, match="filters and columns are ignored"):
        mini_G_bidi.bidirectional_connections(1, **options)


def test_bidirectional_connections_no_edge_info(mini_G_bidi):
    """Test fallback to graph when edge information is unavailable."""
    mini_G_bidi.edge_info = None

    with pytest.warns(
        UserWarning,
        match="No edge information available. "
        "Using graph information instead.",
    ):
        connections = mini_G_bidi.bidirectional_connections(
            {"name": "B"},
            connections_lookup=ConnectionsLookup.ALL,
        )

    assert connections.shape == (4, 3)
