import polars as pl
import polars.testing as pltest
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


@pytest.mark.parametrize(
    ("selection_dict", "expected"),
    [
        pytest.param({"name": "A"}, [0], id="index A"),
        pytest.param({"name": "B"}, [1], id="index B"),
        pytest.param({"name": "C"}, [2], id="index C"),
        pytest.param({"custom_index": 1}, [0], id="custom index 1"),
        pytest.param({"group": "AB"}, [0, 1], id="multiple indices"),
        pytest.param(
            {"name": "A", "group": "AB"}, [0], id="multiple selections"
        ),
    ],
)
def test_find_node_indices(mini_G, selection_dict, expected):
    node_indices = mini_G.find_node_indices(selection_dict)
    assert node_indices == expected


@pytest.mark.parametrize(
    ("selection_dict", "unique", "error"),
    [
        pytest.param(
            {"name": "C", "group": "AB"},
            False,
            KeyError("No node matches"),
            id="no match (multiple selections)",
        ),
        pytest.param(
            {"name": "Z"},
            False,
            KeyError("No node matches"),
            id="no match",
        ),
        pytest.param(
            {"group": "AB"},
            True,
            ValueError("Multiple nodes match"),
            id="multiple matches while unique=True",
        ),
    ],
)
def test_find_node_indices_errors(mini_G, selection_dict, unique, error):
    with pytest.raises(type(error), match=error.args[0]):
        mini_G.find_node_indices(selection_dict, unique=unique)


@pytest.mark.parametrize(
    ("node_indexes", "expected_metadata"),
    [
        pytest.param(
            0,
            pl.DataFrame(
                {
                    "name": ["A"],
                    "group": ["AB"],
                    "notes": ["A is part of group AB"],
                    "custom_index": [1],
                    "node_index": [0],
                }
            ),
            id="single index A",
        ),
        pytest.param(
            [0, 2],
            pl.DataFrame(
                {
                    "name": ["A", "C"],
                    "group": ["AB", "C"],
                    "notes": [
                        "A is part of group AB",
                        "C is part of group C",
                    ],
                    "custom_index": [1, 3],
                    "node_index": [0, 2],
                }
            ),
            id="multiple indices A and C",
        ),
    ],
)
def test_node_metadata_from_indices(mini_G, node_indexes, expected_metadata):
    metadata = mini_G.node_metadata_from_indices(node_indexes)
    pltest.assert_frame_equal(metadata, expected_metadata)


@pytest.mark.parametrize(
    ("node_indexes", "column", "expected_metadata"),
    [
        pytest.param(
            [0, 2],
            "group",
            ["AB", "C"],
            id="Group metadata (multiple indices)",
        ),
        pytest.param(
            [0],
            "name",
            "A",
            id="Name metadata(one index)",
        ),
    ],
)
def test_node_metadata_colum_from_indices(
    mini_G, node_indexes, column, expected_metadata
):
    metadata = mini_G.node_metadata_from_indices(node_indexes, column)
    assert metadata == expected_metadata
