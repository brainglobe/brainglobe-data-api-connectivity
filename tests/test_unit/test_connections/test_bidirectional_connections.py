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

    ALL (edge info):
        A: 2  (A <-> B)
        B: 5  (B <-> A, B <-> C; extra C -> B report)
        C: 5  (C <-> B, C <-> D; extra C -> B report)
        D: 2  (D <-> C)

    """
    return Connections(nodes, bidi_edge_list, bidi_edge_info)


@pytest.mark.parametrize(
    [
        "node",
        "connections_lookup",
        "expected_rows",
    ],
    [
        pytest.param(
            {"name": "A"},
            ConnectionsLookup.ALL,
            2,
            id="A (all)",
        ),
        pytest.param(
            {"name": "A"},
            ConnectionsLookup.NETWORK,
            2,
            id="A (reported)",
        ),
        pytest.param(
            {"name": "B"},
            ConnectionsLookup.ALL,
            5,
            id="B (all)",
        ),
        pytest.param(
            1,
            ConnectionsLookup.ALL,
            5,
            id="B by index (all)",
        ),
        pytest.param(
            {"name": "B"},
            ConnectionsLookup.NETWORK,
            4,
            id="B (reported)",
        ),
        pytest.param(
            {"name": "D"},
            ConnectionsLookup.ALL,
            2,
            id="D (all)",
        ),
        pytest.param(
            {"name": "D"},
            ConnectionsLookup.NETWORK,
            0,
            id="D (reported)",
        ),
    ],
)
def test_bidirectional_connections(
    mini_G_bidi,
    node,
    connections_lookup,
    expected_rows,
):
    """Test finding all bidirectional connections for a node."""
    connections = mini_G_bidi.bidirectional_connections(
        node,
        connections_lookup=connections_lookup,
    )

    assert connections.shape[0] == expected_rows


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

    assert connections.shape[0] == 4
