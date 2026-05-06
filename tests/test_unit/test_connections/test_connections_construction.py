import polars as pl
import pytest
import pytest_mock

from brainglobe_data_api_connectivity.connections import Connections


@pytest.mark.parametrize(
    "existing_node_indexing",
    [
        pytest.param(None, id="No custom indexing"),
        pytest.param("reverse_custom_index", id="Custom index (reversed)"),
    ],
)
def test_connections_setup_network(
    existing_node_indexing: str | None,
    DATA_DIR,
    read_edge_table,
    mocker: pytest_mock.MockerFixture,
    nodes="small-nodes.csv",
    edge_table="small-edge-table.csv",
) -> None:
    """"""
    # Suppress setting of attributes on creation
    mocker.patch.object(Connections, "__init__", lambda *args, **kwargs: None)
    G = Connections()

    _node_file = DATA_DIR / nodes
    _edge_file = DATA_DIR / edge_table

    nodes = pl.read_csv(_node_file)
    edge_table = read_edge_table(_edge_file)

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

        for from_node, to_node, weight in edge_table:
            assert (from_node, to_node) in constructed_edge_list
            assert G.network.get_edge_data(from_node, to_node) == weight
    else:
        # Index translation was necessary. Confirm this was done correctly.
        assert index_translations is not None

        for from_node, to_node, weight in edge_table:
            new_from = index_translations[from_node]
            new_to = index_translations[to_node]
            assert (new_from, new_to) in constructed_edge_list
            assert G.network.get_edge_data(new_from, new_to) == weight


@pytest.mark.parametrize(
    "from_files",
    [
        pytest.param(True, id="From files"),
        pytest.param(False, id="Direct construction"),
    ],
)
@pytest.mark.parametrize(
    ("edge_metadata", "kwargs"),
    [
        pytest.param(None, {}, id="No metadata"),
        pytest.param("small-edge-meta.csv", {}, id="With metadata"),
        pytest.param(
            "small-edge-meta.csv",
            {"edge_meta_from_col": "from_alt", "edge_meta_to_col": "to_alt"},
            id="With metadata, custom to/from columns",
        ),
    ],
)
def test_connections_construction(
    DATA_DIR,
    read_edge_table,
    edge_metadata: str | None,
    kwargs: dict[str, str],
    from_files: bool,
    nodes="small-nodes.csv",
    edge_table="small-edge-table.csv",
) -> None:
    """Tests that, if nodes are provided using a different convention to row
    index identifiers, re-indexing via the network setup is handled correctly.
    """
    _node_file = DATA_DIR / nodes
    _edge_file = DATA_DIR / edge_table
    _edge_meta_file = DATA_DIR / edge_metadata if edge_metadata else None

    nodes = pl.read_csv(_node_file)
    edge_table = read_edge_table(_edge_file)

    meta_before: None | pl.DataFrame
    if edge_metadata is not None:
        edge_metadata = pl.read_csv(_edge_meta_file)
        meta_before = pl.DataFrame(edge_metadata)
    else:
        meta_before = None
    nodes_before = pl.DataFrame(nodes)

    if from_files:
        G = Connections.from_files(
            _node_file, _edge_file, _edge_meta_file, **kwargs
        )
    else:
        G = Connections(edge_table, nodes, edge_metadata, **kwargs)

    nodes_after = G.nodes
    meta_after = G.edge_info

    # Don't drop any nodes...
    assert nodes_after.shape[0] == nodes_before.shape[0]
    assert G.network.num_nodes() == nodes_after.shape[0]
    # ...but add internal index column
    assert nodes_after.shape[1] == nodes_before.shape[1] + 1
    assert G._node_internal_index_col in G.nodes

    # Place all edges into the network
    constructed_edge_list = G.network.edge_list()
    assert len(edge_table) == len(constructed_edge_list)
    for from_node, to_node, weight in edge_table:
        assert (from_node, to_node) in constructed_edge_list
        assert G.network.get_edge_data(from_node, to_node) == weight

    if meta_before is None:
        assert meta_after is None
        assert G.ei_from_col is None
        assert G.ei_to_col is None
    else:
        assert meta_after is not None

        # Identify the columns that would have been set as the from and to
        # columns in the edge metadata table.
        from_col = kwargs.get("edge_meta_from_col", "from")
        to_col = kwargs.get("edge_meta_to_col", "to")

        # Metadata table should retain shape
        assert meta_before.shape == meta_after.shape
        # ...and attributes that identify the from and to columns should be
        # populated
        assert G.ei_from_col == from_col
        assert G.ei_to_col == to_col

        metadata_headers = list(
            c for c in meta_before.columns if c not in {from_col, to_col}
        )
        # All non "to"- and "from"-column entries should have been preserved
        for row in meta_before.iter_rows(named=True):
            identical_row = meta_after.row(
                by_predicate=(
                    (pl.col(G.ei_from_col) == row[from_col])
                    & (pl.col(G.ei_to_col) == row[to_col])
                ),
                named=True,
            )
            for header in metadata_headers:
                assert identical_row[header] == row[header]


def test_connections_construction_same_from_to_cols(
    DATA_DIR,
    read_edge_table,
    raises_error,
    expected_error: Exception = ValueError(
        "Connection metadata 'from' and 'to' columns are the same (to_from)."
    ),
    nodes="small-nodes.csv",
    edge_table="small-edge-table.csv",
    edge_metadata="small-edge-meta.csv",
):
    """Check that one cannot supply the same column as both the 'from' node and
    'to' node when supplying edge metadata.
    """
    edge_table = read_edge_table(DATA_DIR / edge_table)
    nodes = pl.read_csv(DATA_DIR / nodes)
    edge_metadata = pl.read_csv(DATA_DIR / edge_metadata)
    with raises_error(expected_error):
        Connections(
            edge_table,
            nodes,
            edge_metadata,
            edge_meta_from_col="to_from",
            edge_meta_to_col="to_from",
        )
