from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

Cost = TypeVar("Cost")


class DijkstraStrategy(ABC, Generic[Cost]):
    """"""

    weight_fn: Callable[[float], float]

    @property
    def starting_node_initial_distance(self) -> float:
        """Initial distance assigned to the starting node."""
        return self._starting_node_initial_distance()

    @property
    def regular_node_unreached_distance(self) -> float:
        """Distance assigned to un-reached regular nodes."""
        return self._regular_node_unreached_distance()

    @classmethod
    def _starting_node_initial_distance(cls) -> float:
        """Initial distance assigned to the starting node.

        Typically the distance to the starting node is 0 for most
        implementations of Dijkstra's algorithm, which is what the base class
        implements. However this value can be overwritten by subclasses if
        necessary (hence we use a private function, rather than an attribute).
        """
        return 0.0

    @classmethod
    def _regular_node_unreached_distance(cls) -> float:
        """Placeholder value for the distance to as-of-yet un-reached nodes in
        the Dijkstra search.

        Typically this value is set to `inf`, since we always want to accept
        the first path we find to a node as the "current best", even if we
        later find a better one, which is what the base class implements.
        However this value can be overwritten by subclasses if necessary (hence
        we use a private function, rather than an attribute).
        """
        return float("inf")

    def __init__(self, weight_fn=lambda x: x) -> None:
        """"""
        self.weight_fn = weight_fn

    @abstractmethod
    def _distance_to(
        self, current_cost: Cost, next_edge_weight: float
    ) -> Cost:
        """Subclass-specific distance calculation logic.

        Note that `next_edge_weight` has already been passed into
        `self.weight_fn`, before this method was called.
        """

    def distance_to(self, current_cost: Cost, next_edge_weight: float) -> Cost:
        """Distance to the next node, given cost of reaching the current node.

        Note that `self.weight_fn` will be called on `next_edge_weight` prior
        to computing the distance.
        """
        return self._distance_to(
            current_cost, self.weight_fn(next_edge_weight)
        )

    @abstractmethod
    def is_lower_cost(self, current_cost: Cost, proposed_cost: Cost) -> bool:
        """Determine if the proposed cost is better than the current cost.

        Return `True` in the event that the proposed cost is superior to the
        current cost. Typically lowering the cost is preferred, though some
        implementations may choose to employ alternatives.

        This logic must be explicitly implemented by the subclass.
        """
