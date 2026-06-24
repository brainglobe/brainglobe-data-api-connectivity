import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections


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
