from pathlib import Path

from brainglobe_data_api_connectivity.connections import Connections


def test_me(
    DATA_DIR: Path,
    node_file: str = "small-nodes.csv",
    edges_file: str = "small-edges.csv",
) -> None:
    """"""
    node_file = DATA_DIR / node_file
    edges_file = DATA_DIR / edges_file

    G = Connections.from_files(node_file, edges_file)

    pass
