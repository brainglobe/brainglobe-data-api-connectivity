from pathlib import Path

import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.connections.query_opts import (
    ConnectionsLookup,
    NodeInConnection,
)


def test_direct_connections_no_edge_info(
    raises_error,
    expected_error: Exception = TypeError(
        "Edge information is not assigned (`self.edge_info` is `None`)"
    ),
) -> None:
    """
    Test that attempting to query by 'all' connections when edge metadata is
    not set raises an appropriate error.
    """
    G = Connections(
        node_info=pl.DataFrame({"name": ["a", "b", "c"]}),
        edge_table=[(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0)],
        edge_info=None,
    )
    with raises_error(expected_error):
        G.direct_connections(0, connections_lookup=ConnectionsLookup.ALL)


@pytest.mark.parametrize(
    (
        "node",
        "node_as",
        "report_or_all",
        "expected_as_input",
        "expected_as_output",
    ),
    [
        pytest.param(
            2,
            NodeInConnection.EITHER,
            ConnectionsLookup.REPORTED,
            [0, 1, 3],
            [0],
            id="2, either, reported",
        ),
        pytest.param(
            2,
            NodeInConnection.INPUT,
            ConnectionsLookup.REPORTED,
            [0, 1, 3],
            [],
            id="2, input, reported",
        ),
        pytest.param(
            2,
            NodeInConnection.OUTPUT,
            ConnectionsLookup.REPORTED,
            [],
            [0],
            id="2, output, reported",
        ),
        pytest.param(
            2,
            NodeInConnection.EITHER,
            ConnectionsLookup.REPORTED,
            [0, 1, 3],
            [0],
            id="2, either, all",
        ),
        pytest.param(
            2,
            NodeInConnection.INPUT,
            ConnectionsLookup.ALL,
            [0, 1, 3],
            [],
            id="2, input, all",
        ),
        pytest.param(
            2,
            NodeInConnection.OUTPUT,
            ConnectionsLookup.ALL,
            [],
            [0],
            id="2, output, all",
        ),
        pytest.param(
            0,
            NodeInConnection.EITHER,
            ConnectionsLookup.REPORTED,
            [0, 1, 2, 4],
            [0, 1, 2, 3],
            id="Asymmetry between input/output",
        ),
        pytest.param(
            0,
            NodeInConnection.EITHER,
            ConnectionsLookup.ALL,
            [0, 1, 2, 4],
            [0, 1, 2, 3],
            id="Filter duplicates when using 'all'",
        ),
        pytest.param(
            4,
            NodeInConnection.INPUT,
            ConnectionsLookup.ALL,
            [1, 3],
            [],
            id="See edge only in edge info",
        ),
        pytest.param(
            4,
            NodeInConnection.INPUT,
            ConnectionsLookup.REPORTED,
            [1],
            [],
            id="Don't see edge only in edge info",
        ),
    ],
)
def test_direct_connections(
    node: int,
    node_as: NodeInConnection,
    report_or_all: ConnectionsLookup,
    expected_as_input: list[int],
    expected_as_output: list[int],
    DATA_DIR: Path,
    node_info: str = "small-nodes.csv",
    edge_table: str = "small-edge-table.csv",
    edge_info: str = "small-edge-meta.csv",
) -> None:
    """"""
    G = Connections.from_files(
        node_info=DATA_DIR / node_info,
        edge_table=DATA_DIR / edge_table,
        edge_info=DATA_DIR / edge_info,
    )

    as_input, as_output = G.direct_connections(
        node, node_as=node_as, connections_lookup=report_or_all
    )

    # Perform sorting, and cast the expected results to a set first, since this
    # will catch if `direct_connections` isn't removing duplicates.
    assert sorted(set(expected_as_input)) == sorted(as_input)
    assert sorted(set(expected_as_output)) == sorted(as_output)


@pytest.mark.parametrize("report_or_all", ConnectionsLookup)
def test_direct_connections_isolated_node(
    report_or_all: ConnectionsLookup,
) -> None:
    """Test that `direct_connections` correctly handles isolated nodes.

    Note that the small-* csvs don't contain any isolated nodes, so we need to
    build a graph that explicitly includes one, hence the separate test.
    """
    G = Connections(
        node_info=pl.DataFrame({"name": ["a", "b", "c"]}),
        edge_table=[(0, 1, 1.0), (1, 0, 2.0)],
        edge_info=pl.DataFrame(
            {"to": [0, 1], "from": [1, 0], "data": ["1->0", "0->1"]}
        ),
    )
    isolated_node = G.node_indexes_from_information(pl.col("name") == "c")[0]

    as_input, as_output = G.direct_connections(
        isolated_node, connections_lookup=report_or_all
    )
    assert [] == as_input
    assert [] == as_output
