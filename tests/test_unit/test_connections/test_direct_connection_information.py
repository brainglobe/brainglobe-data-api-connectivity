import polars as pl
import pytest
from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity.connections.query_opts import (
    ConnectionsLookup,
    NodeIs,
)


@pytest.mark.parametrize(
    ("node", "node_as", "connections_lookup", "return_edges", "expected"),
    [
        pytest.param(
            1,
            NodeIs.INPUT,
            ConnectionsLookup.REPORTED,
            False,
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
            id="index input nodes",
        ),
        pytest.param(
            1,
            NodeIs.INPUT,
            ConnectionsLookup.ALL,
            True,
            pl.DataFrame(
                {
                    "from_id": ["B"],
                    "to_id": ["C"],
                    "from": [1],
                    "to": [2],
                    "used": ["yes"],
                    "paper": ["author et al., 2025"],
                    "strength": ["medium (1.0)"],
                    "__idx_from": [1],
                    "__idx_to": [2],
                }
            ),
            id="index input all edges",
        ),
        pytest.param(
            {"name": "B"},
            NodeIs.OUTPUT,
            ConnectionsLookup.ALL,
            False,
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
            id="name output nodes",
        ),
        pytest.param(
            {"name": "B"},
            NodeIs.OUTPUT,
            ConnectionsLookup.REPORTED,
            True,
            pl.DataFrame({"from": [0], "to": [1], "value": [0.1]}),
            id="name output reported edges",
        ),
        pytest.param(
            {"name": "B", "group": "AB"},
            NodeIs.ANY,
            ConnectionsLookup.REPORTED,
            False,
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
            id="name and group both any nodes",
        ),
        pytest.param(
            {"name": "B", "group": "AB"},
            NodeIs.ANY,
            ConnectionsLookup.REPORTED,
            True,
            pl.DataFrame({"from": [1, 0], "to": [2, 1], "value": [1.0, 0.1]}),
            id="name and group both any reported edges",
        ),
        pytest.param(
            {"name": "B", "group": "AB"},
            NodeIs.ANY,
            ConnectionsLookup.ALL,
            True,
            pl.DataFrame(
                {
                    "from_id": ["B", "A"],
                    "to_id": ["C", "B"],
                    "from": [1, 0],
                    "to": [2, 1],
                    "used": ["yes", "yes"],
                    "paper": ["author et al., 2025", "author et al., 1998"],
                    "strength": ["medium (1.0)", "weak (0.1)"],
                    "__idx_from": [1, 0],
                    "__idx_to": [2, 1],
                }
            ),
            id="name and group both any all edges",
        ),
    ],
)
def test_direct_connection_info(
    mini_G, node, node_as, connections_lookup, return_edges, expected
) -> None:
    """Return node or edge information from the requested lookup source."""
    result = mini_G.direct_connection_info(
        node,
        node_as,
        connections_lookup=connections_lookup,
        return_edges=return_edges,
    )
    assert_frame_equal(result, expected, check_row_order=not return_edges)


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
