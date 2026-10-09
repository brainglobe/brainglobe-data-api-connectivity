import polars as pl
import pytest
from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity.connections.query_opts import (
    NodeIs,
)


@pytest.mark.parametrize(
    ("node", "node_as", "expected"),
    [
        pytest.param(
            1,
            NodeIs.INPUT,
            pl.DataFrame(
                {
                    "name": ["C"],
                    "idx": [2],
                    "group": ["C"],
                    "notes": ["C is part of group C"],
                    "custom_index": [3],
                    "__node_index": [2],
                    "node_as": ["input"],
                }
            ),
            id="index input",
        ),
        pytest.param(
            {"name": "B"},
            NodeIs.OUTPUT,
            pl.DataFrame(
                {
                    "name": ["A"],
                    "idx": [0],
                    "group": ["AB"],
                    "notes": ["A is part of group AB"],
                    "custom_index": [1],
                    "__node_index": [0],
                    "node_as": ["output"],
                }
            ),
            id="name output",
        ),
        pytest.param(
            {"name": "B", "group": "AB"},
            NodeIs.ANY,
            pl.DataFrame(
                {
                    "name": ["C", "A"],
                    "idx": [2, 0],
                    "group": ["C", "AB"],
                    "notes": ["C is part of group C", "A is part of group AB"],
                    "custom_index": [3, 1],
                    "__node_index": [2, 0],
                    "node_as": ["input", "output"],
                }
            ),
            id="name and group both any",
        ),
    ],
)
def test_direct_connection_info(mini_G, node, node_as, expected) -> None:
    """Return ordered node metadata and roles for the queried node."""
    result = mini_G.direct_connection_info(node, node_as)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("return_edges", "expected"),
    [
        pytest.param(
            False,
            pl.DataFrame(
                schema={
                    "name": pl.String,
                    "idx": pl.Int64,
                    "group": pl.String,
                    "notes": pl.String,
                    "custom_index": pl.Int64,
                    "__node_index": pl.Int64,
                    "node_as": pl.String,
                }
            ),
            id="nodes",
        ),
        pytest.param(True, pl.DataFrame(), id="edges"),
    ],
)
def test_direct_connection_info_empty(mini_G, return_edges, expected) -> None:
    """Retain the output columns and types when there are no connections."""
    mini_G.network.remove_edge(2, 3)  # so that D does not have connections
    result = mini_G.direct_connection_info(
        {"name": "D"}, return_edges=return_edges
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("node", "expected_count"),
    [
        pytest.param({"name": "missing"}, 0, id="invalid (missing)"),
        pytest.param({"group": "AB"}, 2, id="invalid (not unique)"),
    ],
)
def test_direct_connection_info_invalid(mini_G, node, expected_count) -> None:
    """Reject missing or ambiguous identifiers using the backend message."""
    with pytest.raises(
        ValueError, match=f"^Expected 1 unique node, but got {expected_count}"
    ):
        mini_G.direct_connection_info(node)
