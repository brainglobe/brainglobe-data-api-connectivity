from ._base import DijkstraStrategy


class WidestPath(DijkstraStrategy[float]):
    """Strategy that finds the widest path.
    
    The widest path between two regions maximises the _minimum_ edge weight in the path that is chosen. The 'cost' of a path is the minimum edge weight used in that path, and paths with a _greater_ cost are considered better.
    
    The widest path problem can also go by the name of the 'bandwidth problem', because the cost of the widest path is the maximum bandwidth between two regions.
    """

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
