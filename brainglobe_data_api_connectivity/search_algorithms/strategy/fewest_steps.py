from ._base import DijkstraStrategy


class FewestSteps(DijkstraStrategy[float]):
    """Strategy that finds the path with the fewest number of steps.

    Cost increases by 1 for every move, and lower costs are considered better.

    In a network where edges represent connections between brain regions,
    this strategy finds the path that passes through the smallest number of
    regions.
    """

    def _cost_to(self, current_cost: float, next_edge_weight: float) -> float:
        return current_cost + 1

    def is_better_cost(
        self, proposed_cost: float, current_cost: float
    ) -> bool:
        return proposed_cost < current_cost
