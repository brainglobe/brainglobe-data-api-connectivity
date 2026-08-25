from typing import Hashable

import polars as pl
import pytest
import pytest_mock
from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity._types import EdgeTable
from brainglobe_data_api_connectivity.connections import Connections


@pytest.fixture
def abc_nodes() -> pl.DataFrame:
    """Simple collection of nodes to use when checking network setup."""
    return pl.DataFrame(
        {
            "name": ["alpha", "beta", "gamma"],
            "mass": [1.0, 5.0, 10.0],
            "custom_index": [1, 2, 3],
            "rev_custom_index": [3, 2, 1],
        }
    )


@pytest.fixture
def edge_metadata() -> pl.DataFrame:
    "A simple 'edge information' frame for testing setup."
    return pl.DataFrame(
        {
            "to": [0, 1, 2],
            "from": [1, 2, 0],
            "type": ["a", "b", "c"],
        }
    )


@pytest.mark.parametrize(
    (
        "nodes",
        "edge_table",
        "existing_node_indexing",
        "expected_node_indexing",
    ),
    [
        pytest.param(
            pl.DataFrame({"nodes": []}),
            (),
            None,
            None,
            id="Empty graph",
        ),
        pytest.param(
            "abc_nodes",
            [
                (0, 1, 1),  # alpha -> beta, 1
                (0, 2, 2),  # alpha -> gamma, 2
                (1, 0, -1),  # beta -> alpha, -1
                (1, 2, 3),  # beta -> gamma, 3
                (2, 0, -2),  # gamma - alpha, -2
            ],
            None,
            None,
            id="No custom indexing",
        ),
        pytest.param(
            "abc_nodes",
            [
                (1, 2, 1),  # alpha -> beta, 1
                (1, 3, 2),  # alpha -> gamma, 2
                (2, 1, -1),  # beta -> alpha, -1
                (2, 3, 3),  # beta -> gamma, 3
                (3, 1, -2),  # gamma - alpha, -2
            ],
            "custom_index",
            {1: 0, 2: 1, 3: 2},
            id="Custom indexing",
        ),
        pytest.param(
            "abc_nodes",
            [
                (3, 2, 1),  # alpha -> beta, 1
                (3, 1, 2),  # alpha -> gamma, 2
                (2, 3, -1),  # beta -> alpha, -1
                (2, 1, 3),  # beta -> gamma, 3
                (1, 3, -2),  # gamma - alpha, -2
            ],
            "rev_custom_index",
            {3: 0, 2: 1, 1: 2},
            id="Reverse custom indexing",
        ),
        pytest.param(
            "abc_nodes",
            [
                ("alpha", "beta", 1),  # alpha -> beta, 1
                ("alpha", "gamma", 2),  # alpha -> gamma, 2
                ("beta", "alpha", -1),  # beta -> alpha, -1
                ("beta", "gamma", 3),  # beta -> gamma, 3
                ("gamma", "alpha", -2),  # gamma - alpha, -2
            ],
            "name",
            {"alpha": 0, "beta": 1, "gamma": 2},
            id="Index on name (str-valued) column",
        ),
    ],
)
def test_connections_setup_network(
    nodes: pl.DataFrame,
    edge_table: EdgeTable,
    existing_node_indexing: str | None,
    expected_node_indexing: dict[int, int] | None,
    mocker: pytest_mock.MockerFixture,
    request: pytest.FixtureRequest,
) -> None:
    if isinstance(nodes, str):
        nodes = request.getfixturevalue(nodes)

    # Suppress setting of attributes on creation,
    mocker.patch.object(Connections, "__init__", lambda *args, **kwargs: None)
    # which allows us to create an "empty" connections object
    G = Connections()

    nodes_before = pl.DataFrame(nodes)
    index_translations = G._setup_network(
        nodes, edge_table, existing_node_indexing=existing_node_indexing
    )
    nodes_after = G.nodes

    # Don't drop any nodes...
    assert nodes_after.shape[0] == nodes_before.shape[0]
    assert G.network.num_nodes() == nodes_after.shape[0]
    # ...but add internal index column
    assert nodes_after.shape[1] == nodes_before.shape[1] + 1
    assert G._node_internal_index_col in G.nodes

    # Place all edges into the network
    constructed_edge_list = G.network.edge_list()
    if existing_node_indexing is None:
        # No index translations were necessary, so nodes should have just been
        # added using their row-indexes as their internal indexes.
        # Edge table did not need to be adapted.
        assert index_translations is None
        assert G.node_index_column is None

        for from_node, to_node, weight in edge_table:
            assert (from_node, to_node) in constructed_edge_list
            assert G.network.get_edge_data(from_node, to_node) == weight
    else:
        # Index translation was necessary. Confirm this was done correctly.
        assert G.node_index_column == existing_node_indexing
        assert index_translations is not None
        assert index_translations == expected_node_indexing

        for from_node, to_node, weight in edge_table:
            new_from = index_translations[from_node]
            new_to = index_translations[to_node]
            assert (new_from, new_to) in constructed_edge_list
            assert G.network.get_edge_data(new_from, new_to) == weight


@pytest.mark.parametrize(
    ("nodes", "edge_table", "existing_index", "expected_error"),
    [
        pytest.param(
            pl.DataFrame({Connections._node_internal_index_col: [0]}),
            (),
            None,
            ValueError(
                f"Heading '{Connections._node_internal_index_col}' must not "
                "be present in the node metadata, as it is reserved for "
                "internal index referencing"
            ),
            id="Reserved node index column present",
        ),
        pytest.param(
            pl.DataFrame({"name": [0, 1, 2]}),
            (),
            "not_a_col",
            ValueError("Heading not_a_col not present in node metadata."),
            id="Existing column not present in node data.",
        ),
        pytest.param(
            pl.DataFrame({"name": [0, 1, 0]}),
            (),
            "name",
            ValueError(
                "Given node index column contains repeat entries, thus cannot "
                "be used as an index."
            ),
            id="Existing index column doesn't have unique values.",
        ),
    ],
)
def test_connections_setup_network_error_catches(
    nodes: pl.DataFrame,
    edge_table: EdgeTable,
    existing_index: str | None,
    expected_error: Exception,
    mocker: pytest_mock.MockerFixture,
    raises_error,
) -> None:
    mocker.patch.object(Connections, "__init__", lambda *args, **kwargs: None)
    G = Connections()

    with raises_error(expected_error):
        G._setup_network(
            nodes, edge_table, existing_node_indexing=existing_index
        )


@pytest.mark.parametrize(
    ("edge_info", "index_translations", "from_column", "to_column"),
    [
        pytest.param(
            None,
            None,
            "from",
            "to",
            id="No metadata",
        ),
        pytest.param(
            "edge_metadata",
            None,
            "from",
            "to",
            id="At face value",
        ),
        pytest.param(
            "edge_metadata",
            {0: 5, 1: 10, 2: 20},
            "from",
            "to",
            id="Re-indexing has occurred",
        ),
        pytest.param(
            "edge_metadata",
            None,
            "to",  # Note that to mimic alternative columns,
            "from",  # we're passing to as from and vice-versa
            id="Alternative from/to columns used",
        ),
        pytest.param(
            "edge_metadata",
            {0: 5, 1: 10, 2: 20},
            "to",
            "from",
            id="Alt columns and re-indexing",
        ),
    ],
)
def test_connections_setup_edge_metadata(
    edge_info: pl.DataFrame | None,
    index_translations: dict[Hashable, int] | None,
    from_column: str,
    to_column: str,
    mocker: pytest_mock.MockerFixture,
    request: pytest.FixtureRequest,
) -> None:
    if isinstance(edge_info, str):
        edge_info = request.getfixturevalue(edge_info)

    mocker.patch.object(Connections, "__init__", lambda *args, **kwargs: None)
    G = Connections()

    G._setup_edge_metadata(
        edge_info, from_column, to_column, index_translations
    )

    if edge_info is None:
        assert G.edge_info is None
        assert G.edge_info_from_col is None
        assert G.edge_info_to_col is None
    else:
        assert G.edge_info is not None
        assert G.edge_info_from_col == from_column
        assert G.edge_info_to_col == to_column

        assert G.edge_info.shape[0] == edge_info.shape[0]
        # Should insert the internal indexing columns
        assert G.edge_info.shape[1] == edge_info.shape[1] + 2

        # Confirm metadata has been updated to respect any index translations
        for row in edge_info.iter_rows(named=True):
            if index_translations is not None:
                idx_from = index_translations[row[from_column]]
                idx_to = index_translations[row[to_column]]
            else:
                idx_from = row[G.edge_info_from_col]
                idx_to = row[G.edge_info_to_col]

            unique_matching_row = G.edge_info.row(
                by_predicate=(pl.col(G._edge_info_from_index_col) == idx_from)
                & (pl.col(G._edge_info_to_index_col) == idx_to),
                named=True,
            )

            for col in edge_info.columns:
                assert row[col] == unique_matching_row[col]


@pytest.mark.parametrize(
    ("edge_meta", "from_column", "to_column", "expected_exception"),
    [
        pytest.param(
            pl.DataFrame(),
            "to_from",
            "to_from",
            ValueError(
                "Connection metadata 'from' and 'to' columns are the same "
                "(to_from)"
            ),
            id="Same 'to' and 'from' column.",
        )
    ],
)
def test_connections_setup_edge_metadata_errors(
    edge_meta: pl.DataFrame | None,
    from_column: str,
    to_column: str,
    expected_exception: Exception,
    mocker: pytest_mock.MockerFixture,
    raises_error,
    indexing_translations: dict[Hashable, int] | None = None,
) -> None:
    mocker.patch.object(Connections, "__init__", lambda *args, **kwargs: None)
    G = Connections()

    with raises_error(expected_exception):
        G._setup_edge_metadata(
            edge_meta, from_column, to_column, indexing_translations
        )


def test_connections_construction(
    DATA_DIR,
    read_edge_table,
    constructor_kwargs: dict = {},
    nodes="small-nodes.csv",
    edge_table="small-edge-table.csv",
    edge_meta="small-edge-meta.csv",
) -> None:
    """Check that identical instances can be created using either constructor
    method.
    """
    nodes = DATA_DIR / nodes
    edge_table = DATA_DIR / edge_table
    edge_meta = DATA_DIR / edge_meta

    G_from_file = Connections.from_files(
        nodes, edge_table, edge_meta, **constructor_kwargs
    )
    G_via_dataframes = Connections(
        pl.read_csv(nodes),
        read_edge_table(edge_table),
        pl.read_csv(edge_meta),
        **constructor_kwargs,
    )

    assert_frame_equal(G_from_file.nodes, G_via_dataframes.nodes)
    assert G_from_file.network.nodes() == G_via_dataframes.network.nodes()
    assert (
        G_from_file.network.edge_list() == G_via_dataframes.network.edge_list()
    )
    assert_frame_equal(G_from_file.edge_info, G_via_dataframes.edge_info)
