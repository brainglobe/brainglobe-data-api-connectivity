from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from brainglobe_data_api_connectivity._types import EdgeTable


@pytest.fixture(scope="session")
def read_edge_table() -> Callable[[Path], EdgeTable]:
    """Read a .csv file that contains edge table information, in
    (from, to, weight) format without headers.
    """

    def _inner(path: Path) -> EdgeTable:
        return tuple(
            t
            for t in pl.read_csv(path, has_header=False).iter_rows(named=False)
        )

    return _inner
