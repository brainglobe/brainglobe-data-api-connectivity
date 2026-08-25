from collections.abc import Callable
from pathlib import Path
from typing import Concatenate

import polars as pl
import pytest


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Callable[Concatenate[pl.DataFrame, ...], Path]:
    """Create a temporary CSV file from a polars dataframe.

    Usage is `tmp_csv(df, file, *args, **kwargs)` with `file`, `*args`, and
    `**kwargs` passed directly to `df.write_csv`.
    """

    def _inner(df: pl.DataFrame, file, *args, **kwargs) -> Path:
        file_location = tmp_path / file
        df.write_csv(file_location, *args, **kwargs)
        return file_location

    return _inner
