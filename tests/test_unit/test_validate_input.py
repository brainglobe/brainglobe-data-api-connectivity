import re

import numpy as np
import pytest

from brainglobe_data_api_connectivity.io.validate_input import (
    validate_adjacency_matrix,
)


@pytest.mark.parametrize(
    ["matrix", "error"],
    [
        pytest.param(
            np.zeros((4, 4)),
            None,
            id="valid",
        ),
        pytest.param(
            np.zeros((5, 4)),
            ValueError(
                "Adjacency matrix must be square, ",
                "but got 5 rows and 4 columns.",
            ),
            id="Adjacency matrix should be square",
        ),
    ],
)
def test_validate_adjacency_matrix(matrix, error):
    if error is None:
        validate_adjacency_matrix(matrix)
    else:
        with pytest.raises(type(error), match=re.escape(str(error))):
            validate_adjacency_matrix(matrix)
