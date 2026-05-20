"""Validation of input data"""

import pandas as pd


def validate_adjacency_matrix(matrix: pd.DataFrame) -> None:
    """Check whether adjacency matrix is square, raising an error if not."""
    rows, cols = matrix.shape

    if rows != cols:
        error_message = (
            "Adjacency matrix must be square, "
            f"but got {rows} rows and {cols} columns."
        )
        raise ValueError(error_message)
