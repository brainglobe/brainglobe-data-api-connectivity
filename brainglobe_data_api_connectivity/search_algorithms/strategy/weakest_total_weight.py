from ._base import DijkstraStrategy


class WeakestTotalWeight(DijkstraStrategy[float]):
    """Strategy that finds the path with the weakest total weight.

    Cost in this case are determined by adding edge weights, lower totals are
    considered better.

    If edge weights reflect connection strength between brain regions,
    this strategy returns the path and cost of the path with the weakest
    total weight.

    Note that this path does **not** necessarily correspond to the path that
    is functionally weakest. A path with one very weak edge that acts as a
    bottleneck, and several strong edges may have a higher total weight than a
    path whose edges are all moderately weak.
    """

    def _cost_to(self, current_cost: float, next_edge_weight: float) -> float:
        return current_cost + next_edge_weight

    def is_better_cost(
        self, proposed_cost: float, current_cost: float
    ) -> bool:
        return proposed_cost < current_cost
