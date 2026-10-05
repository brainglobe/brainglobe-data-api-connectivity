import pytest
from rustworkx import PyDiGraph

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
            id="average rather than total strength",
        ),
        pytest.param(
            [(0, 2, 10), (0, 1, 9), (1, 2, 9), (2, 3, 1)],
            0,
            3,
            [0, 1, 2, 3],
            19 / 3,
            id="keep a weaker average prefix that wins at the target",
        ),
        pytest.param(
            [(0, 1, 2), (1, 2, 100), (2, 1, 100), (1, 3, 2)],
            0,
            3,
            [0, 1, 3],
            2.0,
            id="exclude strong cycles",
        ),
        pytest.param(
            [(0, 1, -4), (1, 2, -2), (0, 2, -5)],
            0,
            2,
            [0, 1, 2],
            -3.0,
            id="negative strengths",
        ),
        pytest.param(
            [(0, 1, 0)],
            0,
            1,
            [0, 1],
            0.0,
            id="zero strength",
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
            [(0, 1, 8)],
            1,
            0,
            None,
            float("-inf"),
            id="respect direction",
        ),
        pytest.param(
            [(0, 1, 8)],
            0,
            0,
            [0],
            0.0,
            id="same source and target",
        ),
    ],
)
def test_strongest_average_path(
    edges, source, target, expected_path, expected_average
) -> None:
    network = PyDiGraph(multigraph=False)
    network.add_nodes_from(range(4))
    network.add_edges_from(edges)

    path, average = strongest_average_path(network, source, target)

    assert path == expected_path
    assert average == pytest.approx(expected_average)


@pytest.mark.parametrize("source, target", [(4, 0), (0, 4), (4, 4)])
def test_strongest_average_path_unknown_region(source, target) -> None:
    network = PyDiGraph(multigraph=False)
    network.add_nodes_from(range(4))

    with pytest.raises(ValueError, match="must exist in the network"):
        strongest_average_path(network, source, target)
