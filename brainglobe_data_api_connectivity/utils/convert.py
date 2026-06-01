import numpy as np
import pandas as pd


def convert_matrix_to_edge_table(
    matrix: pd.DataFrame, include_zeros: bool = False
) -> np.ndarray:
    """Convert adjacency matrix into an edge table.

    Parameters
    matrix : pandas.DataFrame
        Adjacency matrix.

    include_zeros : bool (default=False)
        - If True: use only the non-zero entries of the matrix
        - If False: use all entries of the matrix

    Returns
    edge_table: numpy.ndarray
        with one row per edge and 3 columns:
            - 1. index of the area from which the edge originates
            - 2. index of the area to which the edge points
            - 3. edge weight
    """
    adjacency_matrix = matrix.to_numpy()
    if include_zeros is False:
        connections = matrix.to_numpy().nonzero()
        weights = adjacency_matrix[connections]
        edge_table = np.column_stack(connections + (weights,))
    else:
        rows, cols = np.indices(adjacency_matrix.shape)
        weights = adjacency_matrix[rows, cols]
        edge_table = np.column_stack(
            (rows.ravel(), cols.ravel(), weights.ravel())
        )
    return edge_table
