from collections.abc import Sequence

import pandas as pd


def validate_matrix(
    matrix: pd.DataFrame, row_ids: Sequence[int], col_ids: Sequence[int]
) -> None:
    """Checks whether matrix has expected number of columns and rows."""
    n_rows = len(row_ids)
    n_cols = len(col_ids)

    if n_rows != n_cols:
        raise ValueError(
            f"There are {n_rows} rows and {n_cols} columns,"
            "expected same number of rows and columns"
        )

    if matrix.shape != (n_rows, n_cols):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match expected "
            f"({n_rows}, {n_cols})."
        )
