import multiprocessing
from pathlib import Path
from time import perf_counter

import pytest

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.search_algorithms import (
    dijkstra,
    strongest_average_path,
)
from brainglobe_data_api_connectivity.search_algorithms.strategy import (
    FewestSteps,
    WeakestTotalWeight,
    WidestPath,
)


@pytest.fixture
def cns2f() -> Connections:
    data_dir = Path("data/CNS2f")
    return Connections.from_files(
        data_dir / "CNS2f_node_info.csv",
        data_dir / "CNS2f_edge_table.csv",
        data_dir / "CNS2f_edge_info.csv",
        node_index_column="region_idx",
        edge_info_from_col="origin_region_idx",
        edge_info_to_col="termination_region_idx",
    )


@pytest.mark.parametrize(
    ("source", "target", "strategy", "expected_cost"),
    [
        pytest.param(0, 1, FewestSteps(), 1, id="0 to 1 fewest steps"),
        pytest.param(
            0, 1, WeakestTotalWeight(), 2, id="0 to 1 weakest total weight"
        ),
        pytest.param(0, 1, WidestPath(), 6, id="0 to 1 widest path"),
        pytest.param(1, 0, FewestSteps(), 1, id="1 to 0 fewest steps"),
        pytest.param(
            1, 0, WeakestTotalWeight(), 2, id="1 to 0 weakest total weight"
        ),
        pytest.param(1, 0, WidestPath(), 6, id="1 to 0 widest path"),
        pytest.param(0, 282, FewestSteps(), 2, id="0 to 282 fewest steps"),
        pytest.param(
            0, 282, WeakestTotalWeight(), 3, id="0 to 282 weakest total weight"
        ),
        pytest.param(0, 282, WidestPath(), 6, id="0 to 282 widest path"),
        pytest.param(65, 732, FewestSteps(), 5, id="65 to 732 shortest path"),
    ],
)
def test_dijkstra_cns2f(
    cns2f, source, target, strategy, expected_cost
) -> None:
    """Check three strategies for direct, reverse, and indirect connections."""
    start = perf_counter()
    path, cost = dijkstra(cns2f.network, source, target, strategy)
    print(
        f"Dijkstra {source} -> {target} took "
        f"{perf_counter() - start:.3f} seconds"
    )

    assert path is not None
    assert path[0] == source
    assert path[-1] == target
    assert cost == expected_cost


def _check_strongest_average_path(network, source, target, max_steps) -> None:
    """Run the search separately so the test can stop it."""
    start = perf_counter()
    path, average = strongest_average_path(
        network, source, target, max_steps=max_steps
    )
    print(
        f"Path search with max_steps={max_steps} took "
        f"{perf_counter() - start:.3f} seconds",
        flush=True,
    )
    print(f"Path: {path}, average strength: {average}")

    assert path is not None
    assert path[0] == source
    assert path[-1] == target
    assert len(path) - 1 <= max_steps


@pytest.mark.parametrize("max_steps", [3, 5])
def test_strongest_average_path_cns2f(cns2f, max_steps) -> None:
    """Time a real path search, failing if it takes more than 30 seconds."""
    # Change these indices to try different regions.
    source = 0
    target = 1
    print(f"Searching for a path from {source} to {target}...", flush=True)
    process = multiprocessing.get_context("spawn").Process(
        target=_check_strongest_average_path,
        args=(cns2f.network, source, target, max_steps),
    )
    process.start()
    try:
        process.join(timeout=90)
        if process.is_alive():
            pytest.fail("Strongest-average path search exceeded 30 seconds")
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        process.close()
