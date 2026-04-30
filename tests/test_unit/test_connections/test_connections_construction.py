import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections


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
    nodes="small-nodes.csv",
    edge_table="small-edge-table.csv",
) -> None:
    """Tests that, if nodes are provided using a different convention to row
    index identifiers, re-indexing via the network setup is handled correctly.
    """
    nodes = pl.read_csv(DATA_DIR / nodes)
    edge_table = read_edge_table(DATA_DIR / edge_table)

    meta_before: None | pl.DataFrame
    if edge_metadata is not None:
        edge_metadata = pl.read_csv(DATA_DIR / edge_metadata)
        meta_before = pl.DataFrame(edge_metadata)
    else:
        meta_before = None
    nodes_before = pl.DataFrame(nodes)

    G = Connections(edge_table, nodes, edge_metadata, **kwargs)

    nodes_after = G.nodes
    meta_after = G.edge_info

    # Don't drop any nodes
    assert nodes_after.shape == nodes_before.shape
    assert G.network.num_nodes() == nodes_after.shape[0]

    # Place all edges into the network
    constructed_edge_list = G.network.edge_list()
    assert len(edge_table) == len(constructed_edge_list)
    for from_node, to_node, weight in edge_table:
        assert (from_node, to_node) in constructed_edge_list
        assert G.network.get_edge_data(from_node, to_node) == weight

    if meta_before is None:
        assert meta_after is None
    else:
        assert meta_after is not None

        # Identify the appropriate columns
        from_col = kwargs.get("edge_meta_from_col", "from")
        to_col = kwargs.get("edge_meta_to_col", "to")

        # For each edge that appeared in the OLD metadata, there should
        # be a corresponding entry in the NEW metadata table...
        assert meta_before.shape[0] == meta_after.shape[0]
        # ... but the "to" and "from" column have been morphed a single column
        assert meta_before.shape[1] == meta_after.shape[1] + 1

        metadata_headers = list(
            c for c in meta_before.columns if c not in {from_col, to_col}
        )
        # All non "to"- and "from"-column entries should have been preserved
        for row in meta_before.iter_rows(named=True):
            seek_new_index = [row[from_col], row[to_col]]
            identical_row = meta_after.row(
                by_predicate=(
                    pl.col(G._edge_meta_index_col) == seek_new_index
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


def test_connections_construction_reserved_header_present(
    raises_error,
    expected_error=ValueError(
        f"Heading '{Connections._edge_meta_index_col}' must not be "
        "present in the edge metadata table, as it is reserved"
        "for internal index referencing."
    ),
) -> None:
    """Catch the case where the user's metadata on the edges contains a
    column that uses the same name as the reserved column name for storing
    edge (i, j) directed pairs.
    """
    nodes = pl.DataFrame(
        {
            "name": [0],
        }
    )
    edge_table = [(0, 0, 0)]
    edge_metadata = pl.DataFrame(
        {"from": [0], "to": [0], Connections._edge_meta_index_col: [(0, 0)]}
    )
    with raises_error(expected_error):
        Connections(edge_table, nodes, edge_meta=edge_metadata)
