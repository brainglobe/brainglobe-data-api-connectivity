from collections.abc import Sequence

import pandas as pd

from brainglobe_data_api_connectivity.io.excel import (
    get_col_ids,
    get_row_ids,
)


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


def check_ids(file_path, sheet_name, data_range, row_with_ids, col_with_ids):
    """Specific to this dataset, checks that row and col ids match and that
    they are a range from 1 to n."""
    row_ids = get_row_ids(file_path, sheet_name, data_range, row_with_ids)
    col_ids = get_col_ids(file_path, sheet_name, data_range, col_with_ids)
    if row_ids != col_ids:
        raise ValueError("Row and column IDs do not match.")

    expected_ids = list(range(1, len(row_ids) + 1))
    if row_ids != expected_ids:
        raise ValueError(
            f"Row IDs {row_ids} do not match expected {expected_ids}."
        )
    return row_ids, col_ids
