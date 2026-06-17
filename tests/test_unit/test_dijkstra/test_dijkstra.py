import pytest
from rustworkx import PyDiGraph

from brainglobe_data_api_connectivity.dijkstra import dijkstra
from brainglobe_data_api_connectivity.dijkstra.strategy import (
    DijkstraStrategy,
    LowestCost,
    WidestPath,
)


@pytest.fixture
def network() -> PyDiGraph:
    """Creates a network on which we can test Dijkstra's algorithm."""
    G = PyDiGraph()
    G.add_edges_from(
        [
            (0, 1, 1.0),
            (0, 5, 5.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 5, 2.0),
            (3, 4, 5.0),
            (4, 1, 3.0),
            (4, 5, 4.0),
            (5, 6, 1.0),
            (6, 4, 1.0),
        ]
    )
    return G


@pytest.mark.parametrize(
    ("source", "target", "strategy", "expected_path", "expected_cost"),
    [
        pytest.param(
            0,
            0,
            LowestCost(),
            [0],
            0.0,
            id="Same start and end (shortest distance)",
        ),
        pytest.param(
            0,
            0,
            WidestPath(),
            [0],
            float("inf"),
            id="Same start and end (widest path)",
        ),
        pytest.param(
            0,
            1,
            LowestCost(),
            [0, 1],
            1.0,
            id="Direct connection 0 -> 1",
        ),
        pytest.param(
            6,
            5,
            LowestCost(),
            [6, 2, 5],
            4.0,
            id="6 -> 5, multiple viable paths",
        ),
        pytest.param(
            1,
            4,
            LowestCost(),
            [1, 2, 5, 6, 4],
            5.0,
            id="1 -> 4, 2-step worse than 4-step",
        ),
    ],
)
def test_dijkstra(
    source: int,
    target: int,
    strategy: DijkstraStrategy,
    expected_path: list[int],
    expected_cost: float,
    network: PyDiGraph,
) -> None:
    """"""
    computed_path, computed_cost = dijkstra(network, source, target, strategy)

    assert computed_cost == pytest.approx(expected_cost)
    assert computed_path == expected_path


def test_dijkstra_no_path(
    network: PyDiGraph,
    raises_error,
    source: int = 5,
    target: int = 0,
    strategy: DijkstraStrategy = LowestCost(),
    expected_error: Exception = Exception("FIXME"),
) -> None:
    """Node 0 in our `network` fixture is always un-reachable from any other
    node.
    """
    with raises_error(expected_error):
        dijkstra(network, source, target, strategy)
