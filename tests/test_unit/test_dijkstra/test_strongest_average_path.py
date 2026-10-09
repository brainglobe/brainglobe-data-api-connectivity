import pytest

from brainglobe_data_api_connectivity.dijkstra import strongest_average_path


@pytest.mark.parametrize(
    ("edges", "source", "target", "expected_path", "expected_average"),
    [
        pytest.param(
            [(0, 1, 8), (1, 2, 6), (0, 2, 5)],
            0,
            2,
            [0, 1, 2],
            7.0,
            id="docstring example",
        ),
        pytest.param(
            [(0, 1, 4), (1, 2, 4), (0, 2, 5)],
            0,
            2,
            [0, 2],
            5.0,
            id="highest average, not total",
        ),
        pytest.param(
            [(0, 2, 10), (0, 1, 9), (1, 2, 9), (2, 3, 1)],
            0,
            3,
            [0, 1, 2, 3],
            19 / 3,
            id="weaker prefix wins",
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
    mini_G, edges, source, target, expected_path, expected_average, max_steps
) -> None:
    network = mini_G.network
    network.clear_edges()
    network.add_edges_from(edges)

    path, average = strongest_average_path(
        network, source, target, max_steps=max_steps
    )

    assert path == expected_path
    assert average == pytest.approx(expected_average)


@pytest.mark.parametrize("source, target", [(4, 0), (0, 4), (4, 4)])
def test_strongest_average_path_unknown_region(mini_G, source, target) -> None:
    with pytest.raises(ValueError, match="must exist in the network"):
        strongest_average_path(mini_G.network, source, target)


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
    mini_G, max_steps, expected_path, expected_average
) -> None:
    """Check that max_steps limits the number of edges."""
    path, average = strongest_average_path(
        mini_G.network, 0, 3, max_steps=max_steps
    )

    assert path == expected_path
    assert average == pytest.approx(expected_average)


def test_strongest_average_path_negative_max_steps(mini_G) -> None:
    with pytest.raises(ValueError, match="max_steps must be non-negative"):
        strongest_average_path(mini_G.network, 0, 1, max_steps=-1)
