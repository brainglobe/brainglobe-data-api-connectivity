import numpy as np
import pandas as pd
import pytest

from brainglobe_data_api_connectivity.utils.convert import (
    convert_matrix_to_edge_table,
)


@pytest.mark.parametrize(
    ["matrix", "expected_edge_table"],
    [
        pytest.param(
            pd.DataFrame(
                [
                    [0, 1, 0],
                    [0, 0, 2],
                    [3, 4, 0],
                ]
            ),
            np.array(
                [
                    [0, 1, 1],
                    [1, 2, 2],
                    [2, 0, 3],
                    [2, 1, 4],
                ]
            ),
            id="3x3 matrix with 4 edges",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
            np.empty((0, 3), dtype=int),
            id="2x2 matrix with no edges",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [0, 7],
                    [0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1, 7],
                ]
            ),
            id="2x2 matrix with single edge",
        ),
    ],
)
def test_convert_matrix_to_edge_table(matrix, expected_edge_table):
    edge_table = convert_matrix_to_edge_table(matrix)
    assert np.array_equal(edge_table, expected_edge_table)
