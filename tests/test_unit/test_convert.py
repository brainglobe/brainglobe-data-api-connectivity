import polars as pl
import pytest
from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity.utils.convert import (
    convert_matrix_to_edge_table,
)


@pytest.mark.parametrize(
    ["matrix", "include_zeros", "region_ids", "expected_edge_table"],
    [
        pytest.param(
            pl.DataFrame(
                [
                    [0, 1, 0],
                    [0, 0, 2],
                    [3, 4, 0],
                ],
                orient="row",
            ),
            False,
            pl.Series(["A", "B", "C"]),
            pl.DataFrame(
                [
                    ["A", "B", 1],
                    ["B", "C", 2],
                    ["C", "A", 3],
                    ["C", "B", 4],
                ],
                orient="row",
                schema=["from", "to", "weight"],
            ),
            id="3x3 matrix with 4 edges + region_ids",
        ),
        pytest.param(
            pl.DataFrame(
                [
                    [0, 1, 0],
                    [0, 0, 2],
                    [3, 4, 0],
                ],
                orient="row",
            ),
            False,
            pl.Series([0, 1, 2]),
            pl.DataFrame(
                [
                    [0, 1, 1],
                    [1, 2, 2],
                    [2, 0, 3],
                    [2, 1, 4],
                ],
                orient="row",
                schema=["from", "to", "weight"],
            ),
            id="3x3 matrix with 4 edges + internal node indices",
        ),
        pytest.param(
            pl.DataFrame([[0, 0], [0, 0]], orient="row"),
            False,
            None,
            pl.DataFrame([], schema=["from", "to", "weight"], orient="row"),
            id="2x2 matrix with no edges",
        ),
        pytest.param(
            pl.DataFrame([[0, 0], [0, 0]], orient="row"),
            True,
            None,
            pl.DataFrame(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                ],
                orient="row",
                schema=["from", "to", "weight"],
            ),
            id="2x2 matrix with no edges (include zeros)",
        ),
        pytest.param(
            pl.DataFrame([[0, 7], [0, 0]], orient="row"),
            False,
            None,
            pl.DataFrame(
                [[0, 1, 7]], orient="row", schema=["from", "to", "weight"]
            ),
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
    assert_frame_equal(edge_table, expected_edge_table, check_dtypes=False)
