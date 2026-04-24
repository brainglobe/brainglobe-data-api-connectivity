import re

import numpy as np
import pytest

from brainglobe_data_api_connectivity.preprocess.validate_data import (
    validate_matrix,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "matrix": np.zeros((4, 4)),
                "row_ids": range(0, 4),
                "col_ids": range(0, 4),
            }
        )
    ],
)
def test_validate_matrix(kwargs):
    validate_matrix(**kwargs)


@pytest.mark.parametrize(
    "kwargs,error_message",
    [
        pytest.param(
            {
                "matrix": np.zeros((4, 4)),
                "row_ids": range(0, 4),
                "col_ids": range(0, 5),
            },
            "expected same number of rows and columns",
        ),
        pytest.param(
            {
                "matrix": np.zeros((4, 4)),
                "row_ids": range(0, 5),
                "col_ids": range(0, 5),
            },
            "Matrix shape (4, 4) does not match expected (5, 5).",
        ),
    ],
)
def test_validate_matrix_value_error(kwargs, error_message):
    with pytest.raises(ValueError, match=re.escape(error_message)):
        validate_matrix(**kwargs)
