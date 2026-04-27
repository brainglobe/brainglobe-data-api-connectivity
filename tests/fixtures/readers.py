from pathlib import Path
from collections.abc import Callable

import pandas as pd
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
            for t in pd.read_csv(path, delimiter=",", header=None).itertuples(
                index=False,
                name=None,
            )
        )

    return _inner
