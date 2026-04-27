import pandas as pd
import pytest

from brainglobe_data_api_connectivity.connections import Connections


@pytest.mark.parametrize(
    ("reindex", "edge_metadata", "kwargs"),
    [
        pytest.param(True, None, {}, id="Re-indexing"),
        pytest.param(
            True, "small-edge-meta.csv", {}, id="Re-indexing, with metadata"
        ),
        pytest.param(
            True,
            "small-edge-meta.csv",
            {"edge_meta_from_col": "from_alt", "edge_meta_to_col": "to_alt"},
            id="Re-indexing, with metadata and custom to/from columns",
        ),
        pytest.param(False, None, {}, id="No re-index"),
        pytest.param(
            False, "small-edge-meta.csv", {}, id="No re-index, with metadata"
        ),
        pytest.param(
            False,
            "small-edge-meta.csv",
            {"edge_meta_from_col": "from_alt", "edge_meta_to_col": "to_alt"},
            id="No re-index, with metadata and custom to/from columns",
        ),
    ],
)
def test_connections_construction(
    DATA_DIR,
    read_edge_table,
    reindex: bool,
    edge_metadata: str | None,
    kwargs: dict[str, str],
    nodes="small-nodes.csv",
    edge_table="small-edge-table.csv",
) -> None:
    """Tests that, if nodes are provided using a different convention to row
    index identifiers, re-indexing via the network setup is handled correctly.
    """
    nodes = pd.read_csv(DATA_DIR / nodes, delimiter=",")
    if reindex:
        # Introduce a non-standard indexing of the input nodes,
        # reflecting that we are not using the row-number as the
        # index.
        nodes.index = reversed(nodes.index)
    edge_table = read_edge_table(DATA_DIR / edge_table)

    meta_before: None | pd.DataFrame
    if edge_metadata is not None:
        edge_metadata = pd.read_csv(DATA_DIR / edge_metadata, delimiter=",")
        meta_before = pd.DataFrame(edge_metadata)
    else:
        meta_before = None
    nodes_before = pd.DataFrame(nodes)
    G = Connections(edge_table, nodes, edge_metadata, **kwargs)
    nodes_after = G.nodes
    meta_after = G.edge_info

    # Don't drop any nodes
    assert nodes_after.shape == nodes_before.shape
    if not reindex:
        assert (nodes_before.index == nodes_after.index).all()

    # Node mapping has taken place appropriately.
    # Indexes may have swapped, but information has been kept
    # consistent
    for i, old_index in enumerate(nodes_before.index):
        new_index = nodes_after.index[i]

        old_row = nodes_before.loc[old_index]
        new_row = nodes_after.loc[new_index]

        pd.testing.assert_series_equal(old_row, new_row, check_names=False)

    # Edges should also map consistently
    assert len(edge_table) == len(G.network.edge_list())
    for old_from, old_to, old_weight in edge_table:
        new_from = nodes_after.index[old_from]
        new_to = nodes_after.index[old_to]
        new_weight = G.network.get_edge_data(new_from, new_to)

        assert old_weight == new_weight

    assert (meta_after is None) == (meta_before is None)
    if meta_before is not None:
        assert meta_after is not None
        # Any edge metadata should have updated consistently, too.

        # Identify the appropriate columns
        from_col = kwargs.get("edge_meta_from_col", "from")
        to_col = kwargs.get("edge_meta_to_col", "to")

        # For each edge that appeared in the OLD metadata, there should
        # be a corresponding entry in the NEW metadata table...
        assert meta_before.shape[0] == meta_after.shape[0]
        # ... but the "to" and "from" column have been morphed into the index
        assert meta_before.shape[1] == meta_after.shape[1] + 2

        metadata_headers = list(
            c for c in meta_before.columns if c not in {from_col, to_col}
        )
        # All non "to"- and "from"-column entries should have been preserved
        for _, row in meta_before.iterrows():
            old_from = row[from_col]
            old_to = row[to_col]

            new_index = (
                nodes_after.index[old_from],
                nodes_after.index[old_to],
            )
            new_row = meta_after.loc[new_index]

            pd.testing.assert_series_equal(
                row[metadata_headers],
                new_row[metadata_headers],
                check_names=False,
            )


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
    nodes = pd.read_csv(DATA_DIR / nodes, delimiter=",")
    edge_metadata = pd.read_csv(DATA_DIR / edge_metadata, delimiter=",")
    with raises_error(expected_error):
        Connections(
            edge_table,
            nodes,
            edge_metadata,
            edge_meta_from_col="to_from",
            edge_meta_to_col="to_from",
        )
