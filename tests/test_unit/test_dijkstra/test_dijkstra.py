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
    G.add_nodes_from(range(7))
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
            (6, 2, 1.0),
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
            5,
            0,
            LowestCost(),
            None,
            LowestCost._regular_node_unreached_cost(),
            id="You shall not path",
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
            id="Direct connection 0 -> 1 (lowest cost)",
        ),
        pytest.param(
            0,
            1,
            WidestPath(),
            [0, 1],
            1.0,
            id="Direct connection 0 -> 1 (widest path)",
        ),
        pytest.param(
            6,
            5,
            LowestCost(),
            [6, 2, 5],
            3.0,
            id="6 -> 5, multiple viable paths (lowest cost)",
        ),
        pytest.param(
            6,
            5,
            WidestPath(),
            [6, 2, 5],
            1.0,
            id="6 -> 5, multiple viable paths (widest path)",
        ),
        pytest.param(
            1,
            4,
            LowestCost(),
            [1, 2, 5, 6, 4],
            5.0,
            id="1 -> 4, 2-step worse than 4-step",
        ),
        pytest.param(
            0,
            5,
            WidestPath(),
            [0, 5],
            5.0,
            id="0 -> 5, strictly wider path",
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
