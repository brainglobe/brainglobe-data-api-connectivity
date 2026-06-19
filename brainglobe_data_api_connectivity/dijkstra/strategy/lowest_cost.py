from ._base import DijkstraStrategy


class LowestCost(DijkstraStrategy[float]):
    """"""

    def _cost_to(self, current_cost: float, next_edge_weight: float) -> float:
        return current_cost + next_edge_weight

    def is_better_cost(
        self, proposed_cost: float, current_cost: float
    ) -> bool:
        return proposed_cost < current_cost
