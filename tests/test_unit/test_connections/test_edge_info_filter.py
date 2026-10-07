import polars as pl
import pytest
from polars.testing import assert_frame_equal


def test_edge_info_filter_options(mini_G) -> None:
    """Return unique values for every column in first-occurrence order."""
    filter_options = mini_G.edge_info_filter_options()

    assert filter_options == {
        "from_id": ["A", "B", "C"],
        "to_id": ["B", "C", "D"],
        "from": [0, 1, 2],
        "to": [1, 2, 3],
        "used": ["yes", "no"],
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
        ],
        "__idx_from": [
            0,
            1,
            2,
        ],
        "__idx_to": [
            1,
            2,
            3,
        ],
    }


@pytest.mark.parametrize(
    ("filters", "expected_n_rows"),
    [
        pytest.param(
            {"strength": "medium (1.0)"},
            2,
            id="strength",
        ),
        pytest.param(
            {"used": "no"},
            1,
            id="not used in network",
        ),
        pytest.param(
            {"paper": "author et al., 2020"},
            2,
            id="paper",
        ),
        pytest.param(
            {"paper": "author et al., 2020", "used": "yes"},
            1,
            id="multiple filters",
        ),
    ],
)
def test_edge_info_filter(
    mini_G,
    filters,
    expected_n_rows,
) -> None:
    """Return filtered edge information and check number of rows."""
    filtered_edge_info = mini_G.edge_info_filter(filters)

    assert filtered_edge_info.shape[0] == expected_n_rows


@pytest.mark.parametrize(
    ("filters", "columns", "expected"),
    [
        pytest.param(
            None,
            ["to_id", "from_id"],
            pl.DataFrame(
                {
                    "to_id": ["B", "C", "C", "C", "D"],
                    "from_id": ["A", "B", "A", "A", "C"],
                }
            ),
            id="columns only",
        ),
        pytest.param(
            {"strength": "medium (1.0)"},
            ["from_id", "to_id", "used"],
            pl.DataFrame(
                {
                    "from_id": ["B", "A"],
                    "to_id": ["C", "C"],
                    "used": ["yes", "no"],
                }
            ),
            id="filter on omitted column",
        ),
        pytest.param(
            {"paper": "author et al., 2020", "used": "yes"},
            ["from_id", "to_id", "strength"],
            pl.DataFrame(
                {
                    "from_id": ["C"],
                    "to_id": ["D"],
                    "strength": ["strong (10.0)"],
                }
            ),
            id="multiple filters and columns",
        ),
    ],
)
def test_edge_info_filter_columns(mini_G, filters, columns, expected) -> None:
    """Return the expected rows and columns in the requested order."""
    filtered_edge_info = mini_G.edge_info_filter(filters, columns=columns)

    assert_frame_equal(filtered_edge_info, expected)


def test_edge_info_filter_options_unavailable(mini_G) -> None:
    """Return an empty mapping when edge information is unavailable."""
    mini_G.edge_info = None
    assert mini_G.edge_info_filter_options() == {}


def test_edge_info_filter_no_edge_info(mini_G) -> None:
    """Warn and return None when edge information is unavailable."""
    mini_G.edge_info = None

    with pytest.warns(
        UserWarning,
        match="No edge information available to filter.",
    ):
        filtered_edge_info = mini_G.edge_info_filter({"used": "yes"})

    assert filtered_edge_info is None
