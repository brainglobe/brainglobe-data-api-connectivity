import pytest
from rustworkx import PyDiGraph

from brainglobe_data_api_connectivity.search_algorithms import (
    strongest_average_path,
)


@pytest.mark.parametrize(
    ("edges", "source", "target", "expected_path", "expected_average"),
    [
        pytest.param(
            [(0, 1, 8), (1, 2, 6), (0, 2, 5)],
            0,
            2,
            [0, 1, 2],
            7.0,
            id="strongest indirect route",
        ),
        pytest.param(
            [(0, 1, 4), (1, 2, 4), (0, 2, 5)],
            0,
            2,
            [0, 2],
            5.0,
            id="strongest direct route",
        ),
        pytest.param(
            [(0, 2, 10), (0, 1, 9), (1, 2, 9), (2, 3, 1)],
            0,
            3,
            [0, 1, 2, 3],
            19 / 3,
            id="best complete route",
        ),
        pytest.param(
            [(0, 1, 2), (1, 2, 100), (2, 1, 100), (1, 3, 2)],
            0,
            3,
            [0, 1, 3],
            2.0,
            id="no repeated nodes",
        ),
        pytest.param(
            [(0, 1, -4), (1, 2, -2), (0, 2, -5)],
            0,
            2,
            [0, 1, 2],
            -3.0,
            id="negative weights",
        ),
        pytest.param(
            [(0, 1, 8)],
            0,
            2,
            None,
            float("-inf"),
            id="unreachable target",
        ),
        pytest.param(
            [(0, 1, 8), (1, 0, 8)],
            0,
            0,
            [0, 1, 0],
            8.0,
            id="feedback loop",
        ),
    ],
)
@pytest.mark.parametrize("max_steps", [None, 3], ids=["unlimited", "3 steps"])
def test_strongest_average_path(
    edges, source, target, expected_path, expected_average, max_steps
) -> None:
    """Test that the route with the highest average edge weight is returned."""
    network = PyDiGraph(multigraph=False)
    network.add_nodes_from(range(4))
    network.add_edges_from(edges)

    path, average = strongest_average_path(
        network, source, target, max_steps=max_steps
    )

    assert path == expected_path
    assert average == pytest.approx(expected_average)


def test_strongest_average_path_unknown_region() -> None:
    """Test that a target node not present in the network raises ValueError."""
    network = PyDiGraph()
    network.add_nodes_from(range(4))

    with pytest.raises(ValueError, match="must exist in the network"):
        strongest_average_path(network, 0, 4)


@pytest.mark.parametrize(
    ("max_steps", "expected_path", "expected_average"),
    [
        pytest.param(0, None, float("-inf"), id="zero steps"),
        pytest.param(1, None, float("-inf"), id="too few steps"),
        pytest.param(2, [0, 2, 3], 10.0, id="two steps"),
        pytest.param(None, [0, 2, 3], 10.0, id="unlimited"),
    ],
)
def test_strongest_average_path_max_steps(
    max_steps, expected_path, expected_average
) -> None:
    """Test that max_steps limits how many edges a route may contain.

    Paths from (0) to (3) in example network:
        (0) ──  5 ──> (1) ──  5 ──> (3)
        (0) ── 10 ──> (2) ── 10 ──> (3)
    """
    network = PyDiGraph(multigraph=False)
    network.add_nodes_from(range(4))
    network.add_edges_from([(0, 1, 5), (1, 3, 5), (0, 2, 10), (2, 3, 10)])

    path, average = strongest_average_path(network, 0, 3, max_steps=max_steps)

    assert path == expected_path
    assert average == pytest.approx(expected_average)


def test_strongest_average_path_negative_max_steps() -> None:
    """Test that a negative maximum number of steps raises ValueError."""
    network = PyDiGraph()
    network.add_nodes_from(range(2))

    with pytest.raises(ValueError, match="max_steps must be non-negative"):
        strongest_average_path(network, 0, 1, max_steps=-1)
