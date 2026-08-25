from collections.abc import Callable
from typing import Iterable

import pytest

from brainglobe_data_api_connectivity.connections.node_contractions import (
    average_of_weights,
    max_of_weights,
    min_of_weights,
    sum_of_weights,
)


@pytest.mark.parametrize(
    ("fn", "input_weights", "expected_weight"),
    [
        pytest.param(
            average_of_weights, [0, 0.5, 1, 1.5], 3.0 / 4.0, id="average"
        ),
        pytest.param(max_of_weights, [0, 0.5, 1, 1.5], 1.5, id="max"),
        pytest.param(min_of_weights, [0, 0.5, 1, 1.5], 0.0, id="min"),
        pytest.param(sum_of_weights, [0, 0.5, 1, 1.5], 3.0, id="sum"),
    ],
)
def test_weight_contraction_functions(
    fn: Callable[..., float],
    input_weights: Iterable[float],
    expected_weight: Iterable[float],
) -> None:
    """
    Short test function to provide coverage for the packaged
    edge-weight-contraction functions.

    This mainly exists to keep test coverage happy, as the majority of
    contraction functions just use Python builtins under the hood. Still,
    it checks that we are using the correct signature, and for more complex
    contraction functions we will want to check the logic is being correctly
    executed.
    """
    assert fn(*input_weights) == expected_weight
