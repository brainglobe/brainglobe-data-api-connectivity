from typing import Container

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from brainglobe_data_api_connectivity.connections import Connections


@pytest.fixture
def node_data() -> pl.DataFrame:
    """Creates a simple DataFrame to use to test node index lookups."""
    return pl.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "number": [1, 2, 3, 4],
            "mass": [1.0, 2.0, 3.0, 4.0],
            "manual_index": [100, 75, 50, 25],
        }
    )


@pytest.mark.parametrize(
    ("predicates", "constraints", "expected_indexes"),
    [
        pytest.param([], {}, [100, 75, 50, 25], id="No filters applied"),
        pytest.param([], {"name": "b"}, [75], id="Select only 'b'"),
        pytest.param(
            [pl.col("number") < 3], {}, [100, 75], id="Number less than 3"
        ),
        pytest.param(
            [pl.col("number") < 4, pl.col("mass") > 1.5],
            {"name": "c"},
            [50],
            id="Compound selection",
        ),
    ],
)
def test_node_index_lookup(
    node_data: pl.DataFrame,
    predicates,
    constraints,
    expected_indexes: list[int],
) -> None:
    """Test lookup of graph-node indexes via node metadata."""
    G = Connections(node_data, [])
    # Overwrite the assigned "indexes" from the graph with our own manual
    # indexes, so we can confirm lookup behaviour
    G.nodes.drop_in_place(G._node_internal_index_col)
    G.nodes = G.nodes.rename({"manual_index": G._node_internal_index_col})

    # Run the filter and check that all expected indexes are returned.
    # Order of the rows should also be preserved, so may as well check this
    # holds up too.
    assert (
        expected_indexes
        == G._node_indexes_from_information(
            *predicates, **constraints
        ).to_list()
    )


@pytest.mark.parametrize(
    ("seek_indexes", "should_find_names"),
    [
        pytest.param([100, 50], ["a", "c"], id="Find a & c"),
        pytest.param([], [], id="Seek nothing"),
        pytest.param([13], [], id="Index not used"),
        pytest.param([25, 0], ["d"], id="Some unused indexes"),
    ],
)
def test_node_information_lookup(
    node_data: pl.DataFrame,
    seek_indexes: Container[int],
    should_find_names: Container[str],
) -> None:
    """Very basic test to confirm that information lookup by internal index
    works as intended.

    All this function does is wrap built-in polars functionality, so we just
    run a few bare-minimum checks to confirm everything is as expected.
    """
    G = Connections(node_data, [])
    # Overwrite the assigned "indexes" from the graph with our own manual
    # indexes, so we can confirm lookup behaviour
    G.nodes.drop_in_place(G._node_internal_index_col)
    G.nodes = G.nodes.rename({"manual_index": G._node_internal_index_col})

    # Run the lookup, checking we found the correct names (quick / naive check)
    lookup = G._node_information_from_index(seek_indexes)
    assert_series_equal(
        lookup.get_column("name"),
        pl.Series(should_find_names),
        check_names=False,
        check_dtypes=False,
    )
    # (Overkill) check that we are running the correct filter query internally.
    assert_frame_equal(
        lookup,
        G.nodes.filter(pl.col(G._node_internal_index_col).is_in(seek_indexes)),
    )
