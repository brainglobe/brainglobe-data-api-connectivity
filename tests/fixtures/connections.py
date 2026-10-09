import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections


@pytest.fixture
def mini_G() -> Connections:
    """Four nodes: A -> B -> C -> D, with an additional A -> C edge."""
    return Connections(
        node_info=pl.DataFrame({"name": ["A", "B", "C", "D"]}),
        edge_table=[(0, 1, 0.1), (1, 2, 1.0), (0, 2, 10.0), (2, 3, 10.0)],
    )
