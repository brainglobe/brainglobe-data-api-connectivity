import pytest
from rustworkx import PyDiGraph

from brainglobe_data_api_connectivity.dijkstra import dijkstra
from brainglobe_data_api_connectivity.dijkstra.strategy import (
    DijkstraStrategy,
    FewestSteps,
    LowestCost,
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
            LowestCost(),
            [0, 3, 2],
            3.0,
            id="lowest cost",
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
            LowestCost(),
            0.0,
            id="lowest cost: zero cost",
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
            LowestCost(),
            float("inf"),
            id="lowest cost: zero cost",
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
    ("strategy", "expected_path", "expected_cost"),
    [
        pytest.param(
            WidestPath(),
            [0, 1, 2, 3],
            1.0,
            id="widest path",
        ),
    ],
)
def test_dijkstra_widest_path_cycle(
    strategy: DijkstraStrategy,
    expected_path: list[int],
    expected_cost: float,
):
    """Test that cycles do not affect widest path.

    Graph with a cycle:

        (0) ── 1.0 ──► (1) ── 1.0 ──► (2) ── 1.0 ──► (3)
                        ▲              │
                        └───── 10.0 ───┘

    """
    cycle_network = PyDiGraph()
    cycle_network.add_nodes_from(range(4))
    cycle_network.add_edges_from(
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 1, 10.0),
            (2, 3, 1.0),
        ]
    )
    source = 0
    target = 3
    path, cost = dijkstra(cycle_network, source, target, strategy)
    assert path == expected_path
    assert cost == expected_cost


@pytest.mark.parametrize(
    ("strategy", "expected_path", "expected_cost"),
    [
        pytest.param(
            LowestCost(),
            [0, 1, 2, 3],
            1002.0,
            id="widest path",
        ),
    ],
)
def test_dijkstra_lowest_cost_cycle(
    strategy: DijkstraStrategy,
    expected_path: list[int],
    expected_cost: float,
):
    """Test that cycles do not affect lowest cost

    Graph with a cycle:

        (0) ── 1.0 ──► (1) ── 1.0 ──► (2) ── 1000.0 ──► (3)
                        ▲              │
                        └───── 0.01 ───┘

    """
    cycle_network = PyDiGraph()
    cycle_network.add_nodes_from(range(4))
    cycle_network.add_edges_from(
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 1, 0.01),
            (2, 3, 1000.0),
        ]
    )
    source = 0
    target = 3
    path, cost = dijkstra(cycle_network, source, target, strategy)
    assert path == expected_path
    assert cost == expected_cost
