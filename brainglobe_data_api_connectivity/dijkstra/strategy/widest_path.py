from ._base import Cost, DijkstraStrategy


class WidestPath(DijkstraStrategy[Cost]):
    """"""

    @classmethod
    def _starting_node_initial_distance(cls):
        return super()._starting_node_initial_distance()

    @classmethod
    def _regular_node_unreached_distance(cls):
        return super()._regular_node_unreached_distance()

    def _distance_to(self, current_cost, next_edge_weight):
        return super()._distance_to(current_cost, next_edge_weight)

    def is_lower_cost(self, current_cost, proposed_cost):
        return super().is_lower_cost(current_cost, proposed_cost)
