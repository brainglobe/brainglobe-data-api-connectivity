import numpy as np
import pandas as pd
import pytest

from brainglobe_data_api_connectivity.utils.convert import (
    convert_matrix_to_edge_table,
    lookup_and_morph_node_index,
)

morph_dict = {"one": 1, "two": 2}


def _test_morph(region_side: str) -> int:
    return morph_dict[region_side]


@pytest.mark.parametrize(
    ["info2morph", "region_id", "error", "error_message", "expected_idx"],
    [
        pytest.param("one", "Area1", None, 0),
        pytest.param("two", "Area1", None, 1),
        pytest.param("one", "Area2", None, 2),
        pytest.param(
            "two",
            "Area2",
            ValueError,
            "Found 0 matches, expected 1.",
            None,
        ),
    ],
)
def test_lookup_node_index(
    info2morph, region_id, error, error_message, expected_idx
):

    info = pd.DataFrame(
        {
            "info_to_morph": [1, 2, 1],
            "region_id": ["Area1", "Area1", "Area2"],
        }
    )
    region_info = {"info_to_morph": info2morph, "region_id": region_id}

    region_to_node_heading = {"info": "info_to_morph", "id": "region_id"}
    morph_value = {"info_to_morph": _test_morph}

    if error:
        with pytest.raises(error, match=error_message):
            idx = lookup_and_morph_node_index(
                region_info, info, region_to_node_heading, morph_value
            )
    else:
        idx = lookup_and_morph_node_index(
            region_info, info, region_to_node_heading, morph_value
        )
        assert idx == expected_idx


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
