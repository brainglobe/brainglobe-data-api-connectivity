from ._base import DijkstraStrategy


class WidestPath(DijkstraStrategy[float]):
    """"""

    @classmethod
    def _starting_node_initial_cost(cls):
        return float("inf")

    @classmethod
    def _regular_node_unreached_cost(cls):
        return 0.0

    def _cost_to(self, current_cost: float, next_edge_weight: float) -> float:
        return min(current_cost, next_edge_weight)

    def is_better_cost(
        self, proposed_cost: float, current_cost: float
    ) -> bool:
        return proposed_cost > current_cost
