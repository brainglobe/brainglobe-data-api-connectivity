"""Validation of input data"""

import pandas as pd


def validate_adjacency_matrix(matrix: pd.DataFrame) -> None:
    """Checks whether matrix is square and raises a clear error if not."""
    rows, cols = matrix.shape

    if rows != cols:
        error_message = (
            "Adjacency matrix must be square, "
            f"but got {rows} rows and {cols} columns."
        )
        raise ValueError(error_message)
