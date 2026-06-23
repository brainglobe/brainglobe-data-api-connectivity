import pytest
from rustworkx import PyDiGraph

from brainglobe_data_api_connectivity.dijkstra import dijkstra
from brainglobe_data_api_connectivity.dijkstra.strategy import (
    DijkstraStrategy,
    FewestSteps,
    WeakestTotalWeight,
    WidestPath,
)


@pytest.fixture
def simple_network() -> PyDiGraph:
    """Creates a simple network on which we can test Dijkstra's algorithm.

    Representation of the simple network with nodes (0), (1), (2), and (3)::
    (3) ◀── 2.0 ── (0) ── 5.0 ──▶ (1)
                    │               │
                   4.0              │
                    │               │
                    ▼               │
                    (2) ◀───────── 5.0

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
            WeakestTotalWeight(),
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
            WeakestTotalWeight(),
            0.0,
            id="weakest total weight path: zero cost",
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
            WeakestTotalWeight(),
            float("inf"),
            id="weakest total weight path: zero cost",
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
            WeakestTotalWeight(),
            [
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 1, 0.01),  # "low cost" cycle
                (2, 3, 1000.0),  # "high cost" last step
            ],
            [0, 1, 2, 3],
            1002.0,
            "'weak weight' cycle",
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
