from ._base import DijkstraStrategy


class ShortestDistance(DijkstraStrategy[float]):
    """"""

    def _distance_to(
        self, current_cost: float, next_edge_weight: float
    ) -> float:
        return 0.0

    def is_lower_cost(self, current_cost: float, proposed_cost: float) -> bool:
        return False
