from rustworkx import PyDiGraph

from .strategy import DijkstraStrategy


def dijkstra(
    network: PyDiGraph, source: int, target: int, strategy: DijkstraStrategy
) -> list[int]:
    """"""
