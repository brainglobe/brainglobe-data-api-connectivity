import pandas as pd
import pytest

from brainglobe_data_api_connectivity.utils.convert import (
    convert_matrix_to_edge_table,
)


@pytest.mark.parametrize(
    ["matrix", "include_zeros", "region_ids", "expected_edge_table"],
    [
        pytest.param(
            pd.DataFrame(
                [
                    [0, 1, 0],
                    [0, 0, 2],
                    [3, 4, 0],
                ]
            ),
            False,
            pd.Series(["A", "B", "C"]),
            pd.DataFrame(
                [
                    ["A", "B", 1],
                    ["B", "C", 2],
                    ["C", "A", 3],
                    ["C", "B", 4],
                ]
            ),
            id="3x3 matrix with 4 edges + region_ids",
        ),
        pytest.param(
            pd.DataFrame([[0, 0], [0, 0]]),
            False,
            None,
            pd.DataFrame([], columns=[0, 1, 2]),
            id="2x2 matrix with no edges",
        ),
        pytest.param(
            pd.DataFrame([[0, 0], [0, 0]]),
            True,
            None,
            pd.DataFrame(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                ]
            ),
            id="2x2 matrix with no edges (include zeros)",
        ),
        pytest.param(
            pd.DataFrame([[0, 7], [0, 0]]),
            False,
            None,
            pd.DataFrame([[0, 1, 7]]),
            id="2x2 matrix with single edge",
        ),
    ],
)
def test_convert_matrix_to_edge_table(
    matrix, include_zeros, region_ids, expected_edge_table
):
    edge_table = convert_matrix_to_edge_table(
        matrix, include_zeros, region_ids
    )
    pd.testing.assert_frame_equal(
        edge_table, expected_edge_table, check_dtype=False
    )
