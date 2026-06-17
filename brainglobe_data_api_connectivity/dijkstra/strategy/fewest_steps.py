from ._base import DijkstraStrategy


class FewestSteps(DijkstraStrategy[float]):
    """"""

    def _cost_to(self, current_cost: float, next_edge_weight: float) -> float:
        return current_cost + 1

    def is_better_cost(
        self, current_cost: float, proposed_cost: float
    ) -> bool:
        return current_cost > proposed_cost
