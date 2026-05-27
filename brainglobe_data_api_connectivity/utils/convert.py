import numpy as np
import pandas as pd


def convert_matrix_to_edge_table(matrix: pd.DataFrame) -> np.ndarray:
    """Convert adjacency matrix into an edge table.

    Only non‑zero entries in are interpreted as edges.

    Parameters
    matrix : pandas.DataFrame
        Adjacency matrix.

    Returns
    edge_table: numpy.ndarray
        with one row per edge and 3 columns:
            - 1. index of the area from which the edge originates
            - 2. index of the area to which the edge points
            - 3. edge weight
    """
    adjacency_matrix = matrix.to_numpy()
    connections = adjacency_matrix.nonzero()
    weights = adjacency_matrix[connections]
    edge_table = np.column_stack(connections + (weights,))
    return edge_table
