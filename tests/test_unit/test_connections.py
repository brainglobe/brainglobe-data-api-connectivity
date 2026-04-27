from pathlib import Path

from brainglobe_data_api_connectivity.connections import Connections


def test_me(
    DATA_DIR: Path,
    node_file: Path | str = "small-nodes.csv",
    edge_table: Path | str = "small-edge-table.csv",
    edge_meta: Path | str = "small-edge-meta.csv",
) -> None:
    """"""
    node_file = DATA_DIR / node_file
    edge_table = DATA_DIR / edge_table
    edge_meta = DATA_DIR / edge_meta

    G = Connections.from_files(node_file, edge_table, edge_meta)

    pass
