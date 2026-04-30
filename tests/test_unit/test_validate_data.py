import re

import numpy as np
import pytest

from brainglobe_data_api_connectivity.io.validate_input import (
    validate_matrix,
)


@pytest.mark.parametrize(
    "matrix,row_ids,col_ids,error",
    [
        pytest.param(
            np.zeros((4, 4)),
            range(0, 4),
            range(0, 4),
            None,
            id="valid",
        ),
        pytest.param(
            np.zeros((4, 4)),
            [
                0,
                1,
                2,
                3,
            ],
            [5, 6, 7, 100],
            None,
            id="valid (col_ids!=row_ids)",
        ),
        pytest.param(
            np.zeros((4, 4)),
            range(0, 4),
            range(0, 5),
            ValueError("expected same number of rows and columns"),
            id="n_cols!=n_rows",
        ),
        pytest.param(
            np.zeros((4, 4)),
            range(0, 5),
            range(0, 5),
            ValueError("Matrix shape (4, 4) does not match expected (5, 5)."),
            id="matrix shape does not match expectations",
        ),
    ],
)
def test_validate_matrix_value(matrix, row_ids, col_ids, error):
    if error is None:
        validate_matrix(matrix, row_ids, col_ids)
    else:
        with pytest.raises(type(error), match=re.escape(str(error))):
            validate_matrix(matrix, row_ids, col_ids)
