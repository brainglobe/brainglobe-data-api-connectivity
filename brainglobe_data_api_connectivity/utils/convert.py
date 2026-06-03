import numpy as np
import pandas as pd


def convert_matrix_to_edge_table(
    matrix: pd.DataFrame,
    include_zeros: bool = False,
    region_ids=pd.Series | None,
) -> pd.DataFrame:
    """Convert adjacency matrix into an edge table.

    Parameters
    matrix : pandas.DataFrame
        Adjacency matrix.

    include_zeros : bool (default=False)
        - If True: use only the non-zero entries of the matrix
        - If False: use all entries of the matrix

    region_ids : pandas.Series or None
        Optional mapping from matrix indices to region identifiers.
        If provided, the source and target columns are mapped to these labels.

    Returns
    edge_table : pandas.DataFrame
        Table with one row per edge and 3 columns:
            - 1. region identifier (or index) from which the edge originates
            - 2. region identifier (or index) to which the edge points
            - 3. edge weight
    """
    adjacency_matrix = matrix.to_numpy()
    if include_zeros is False:
        connections = adjacency_matrix.nonzero()
        weights = adjacency_matrix[connections]
        edge_table = np.column_stack(connections + (weights,))
    else:
        rows, cols = np.indices(adjacency_matrix.shape)
        weights = adjacency_matrix[rows, cols]
        edge_table = np.column_stack(
            (rows.ravel(), cols.ravel(), weights.ravel())
        )

    edge_table_df = pd.DataFrame(edge_table)

    if region_ids is not None:
        for col in [0, 1]:
            edge_table_df[col] = edge_table_df[col].map(region_ids)

    return edge_table_df.reset_index(drop=True)
