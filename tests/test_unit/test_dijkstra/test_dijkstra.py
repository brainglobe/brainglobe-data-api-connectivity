import pytest
from rustworkx import PyDiGraph

from brainglobe_data_api_connectivity.dijkstra import dijkstra
from brainglobe_data_api_connectivity.dijkstra.strategy import (
    DijkstraStrategy,
    FewestSteps,
    WeakestPath,
    WidestPath,
)

# TODO: write tests per strategy


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
            WeakestPath(),
            [0],
            0.0,
            id="Same start and end (shortest distance)",
        ),
        pytest.param(
            5,
            0,
            WeakestPath(),
            None,
            WeakestPath._regular_node_unreached_cost(),
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
            WeakestPath(),
            [0, 1],
            1.0,
            id="Direct connection 0 -> 1 (weakest path)",
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
            WeakestPath(),
            [6, 2, 5],
            3.0,
            id="6 -> 5, multiple viable paths (weakest path)",
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
            WeakestPath(),
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


@pytest.fixture
def simple_network() -> PyDiGraph:
    """Creates a simple network on which we can test Dijkstra's algorithm.

    The network includes the following routes from node 0 to node 2:
        - (0) ── 4.0 ──► (2)
        - (0) ── 5.0 ──► (1) ── 5.0 ──► (2)
        - (0) ── 1.0 ──► (3) ── 2.0 ──► (2)
    """
    G = PyDiGraph()
    G.add_nodes_from(range(4))
    G.add_edges_from(
        [
            (0, 2, 4.0),
            (0, 1, 5.0),
            (1, 2, 5.0),
            (0, 3, 1.0),
            (3, 2, 2.0),
        ]
    )
    return G


@pytest.mark.parametrize(
    ("strategy", "expected_path", "expected_cost"),
    [
        pytest.param(
            WeakestPath(),
            [0, 3, 2],
            3.0,
            id="weakest path",
        ),
        pytest.param(
            WidestPath(),
            [0, 1, 2],
            5.0,
            id="widest path",
        ),
        pytest.param(
            FewestSteps(),
            [0, 2],
            1.0,
            id="fewest steps",
        ),
    ],
)
def test_dijkstra_strategies_0_to_2(
    simple_network: PyDiGraph,
    strategy: DijkstraStrategy,
    expected_path: list[int],
    expected_cost: float,
):
    """Test returned path and cost when source is 0 and target is 2."""
    source = 0
    target = 2
    path, cost = dijkstra(simple_network, source, target, strategy)
    assert path == expected_path
    assert cost == expected_cost


@pytest.mark.parametrize(
    ("strategy", "expected_cost"),
    [
        pytest.param(
            WeakestPath(),
            0.0,
            id="weakest path: zero cost",
        ),
        pytest.param(
            WidestPath(),
            float("inf"),
            id="widest path: infinite cost",
        ),
        pytest.param(
            FewestSteps(),
            0.0,
            id="fewest steps: zero cost",
        ),
    ],
)
def test_dijkstra_strategies_same_source_and_target(
    simple_network: PyDiGraph,
    strategy: DijkstraStrategy,
    expected_cost: float,
):
    """Test returned path and cost when source equals target."""

    node = 1
    path, cost = dijkstra(simple_network, node, node, strategy)

    assert path == [node]
    assert cost == strategy._starting_node_initial_cost()
    assert cost == expected_cost


@pytest.mark.parametrize(
    ("strategy", "expected_cost"),
    [
        pytest.param(
            WeakestPath(),
            float("inf"),
            id="weakest path: zero cost",
        ),
        pytest.param(
            WidestPath(),
            0.0,
            id="widest path: infinite cost",
        ),
        pytest.param(
            FewestSteps(),
            float("inf"),
            id="fewest steps: zero cost",
        ),
    ],
)
def test_dijkstra_strategies_unreachable_target(
    simple_network: PyDiGraph,
    strategy: DijkstraStrategy,
    expected_cost: float,
):
    """Test returned path and cost when source equals target."""

    source = 2
    target = 0
    path, cost = dijkstra(simple_network, source, target, strategy)

    assert path is None
    assert cost == strategy._regular_node_unreached_cost()
    assert cost == expected_cost


@pytest.mark.parametrize(
    ("strategy", "edges", "expected_path", "expected_cost", "id"),
    [
        pytest.param(
            WidestPath(),
            [
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 1, 10.0),  # "wide" cycle
                (2, 3, 1.0),
            ],
            [0, 1, 2, 3],
            1.0,
            "'wide' cycle",
        ),
        pytest.param(
            WeakestPath(),
            [
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 1, 0.01),  # "low cost" cycle
                (2, 3, 1000.0),  # "high cost" last step
            ],
            [0, 1, 2, 3],
            1002.0,
            "'low cost' cycle",
        ),
    ],
)
def test_dijkstra_cycle_handling(
    strategy: DijkstraStrategy,
    edges: list[tuple[int, int, float]],
    expected_path: list[int],
    expected_cost: float,
    id: str,
):
    """Test that cycles do not affect path and cost.

    Example cycle structure:

        (0) ──► (1) ──► (2) ──► (3)
                 ▲        │
                 └────────┘
    """
    network = PyDiGraph()
    network.add_nodes_from(range(4))
    network.add_edges_from(edges)

    path, cost = dijkstra(network, 0, 3, strategy)
    assert path == expected_path
    assert cost == expected_cost
