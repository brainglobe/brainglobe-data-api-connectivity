import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.connections.query_opts import NodeIs


@pytest.fixture
def nodes() -> pl.DataFrame:
    """Simple collection of nodes to use when checking network setup."""
    return pl.DataFrame(
        {
            "name": ["A", "B", "C"],
            "group": ["AB", "AB", "C"],
            "notes": [
                "A is part of group AB",
                "B is part of group AB",
                "C is part of group C",
            ],
            "custom_index": [1, 2, 3],
        }
    )


@pytest.fixture
def edge_list() -> list[tuple[int, int, float]]:
    """Edge list fixture compatible with Connections().

    Representation of the simple network with nodes (0), (1), and (2):

    (A) ── 0.1 ──▶ (B) ── 10.0 ──▶ (C)
     │                               ▲
     │                               │
     └───────────── 1.0 ─────────────┘

    """
    return [
        (0, 1, 0.1),
        (1, 2, 1.0),
        (0, 2, 10),
    ]


@pytest.fixture
def edge_info() -> pl.DataFrame:
    "A simple 'edge information' frame for testing setup."
    return pl.DataFrame(
        {
            "from": ["A", "B", "A", "A"],
            "to": ["B", "C", "C", "C"],
            "source_node_idx": [0, 1, 0, 0],  # not always present
            "target_node_idx": [1, 2, 2, 2],  # not always present
            "used": ["yes", "yes", "yes", "no"],
            "paper": [
                "author et al., 1998",
                "author et al., 2025",
                "author et al., 2010",
                "author et al., 2020",
            ],
            "strength": [
                "weak (0.1)",
                "medium (1.0)",
                "strong (10.0)",
                "medium (1.0)",
            ],
        }
    )


@pytest.fixture
def mini_G(nodes, edge_list, edge_info) -> Connections:
    """"""
    return Connections(nodes, edge_list, edge_info)


@pytest.mark.parametrize(
    ["node0", "node1", "node0_as", "expected_bool", "expected_shape"],
    [
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            None,
            True,
            (1, 7),
            id="A to B",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            None,
            True,
            (2, 7),
            id="A to C",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            None,
            True,
            (1, 7),
            id="B to A",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.INPUT,
            False,
            (0, 7),
            id="B to A input only",
        ),
    ],
)
def test_has_direct_connection_between(
    mini_G, node0, node1, node0_as, expected_bool, expected_shape
):
    has_connection, connections = mini_G.direct_connection_between(
        node0,
        node1,
        node0_as=node0_as,
    )

    assert has_connection == expected_bool
    assert connections.shape == expected_shape
