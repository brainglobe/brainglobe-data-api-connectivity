from ._base import Cost, DijkstraStrategy


class ShortestDistance(DijkstraStrategy[Cost]):
    """"""

    def _distance_to(
        self, current_cost: Cost, next_edge_weight: float
    ) -> float:
        return 0.0

    def is_lower_cost(self, current_cost: Cost, proposed_cost: Cost) -> bool:
        return False
