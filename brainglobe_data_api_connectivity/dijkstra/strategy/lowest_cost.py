from ._base import DijkstraStrategy


class LowestCost(DijkstraStrategy[float]):
    """Strategy that finds the path with the lowest cost.

    Costs is determined by adding edge weights, lower totals are considered
    better.

    If edge weights reflect connection strength between brain regions,
    this strategy returns the path and cost of the path with the weakest
    total strength.
    """

    def _cost_to(self, current_cost: float, next_edge_weight: float) -> float:
        return current_cost + next_edge_weight

    def is_better_cost(
        self, proposed_cost: float, current_cost: float
    ) -> bool:
        return proposed_cost < current_cost
