from functools import cmp_to_key

from rustworkx import PyDiGraph

from .strategy import DijkstraStrategy


def _compare_nodes(
    node1, node2, node_best_current_cost, strategy: DijkstraStrategy
):
    """Compare two nodes based on their current best cost.

    `node_best_current_cost` maps each node to the best cost found so far for
    reaching it. The strategy determines which cost is considered better.

    Returns:
    1:  if node1 has a better cost than node2
    -1:  of node1 does not have a better cost than node2

    """
    cost_node1 = node_best_current_cost[node1]
    cost_node2 = node_best_current_cost[node2]
    return 1 if strategy.is_better_cost(cost_node1, cost_node2) else -1


def reconstruct_path(
    path_map: dict[int, int], starting_node: int, destination_node: int
) -> list[int]:
    """Reconstruct path from source to target the `path_map` dict.

    The `path_map` dict maps each reached node to the node from which it
    was reached with the best cost.
    """
    path_list = [destination_node]
    current_node = destination_node
    while path_map[current_node] != starting_node:
        current_node = path_map[current_node]
        path_list.append(current_node)
    path_list.append(starting_node)
    path_list.reverse()
    return path_list


def dijkstra(
    network: PyDiGraph, source: int, target: int, strategy: DijkstraStrategy
) -> tuple[list[int] | None, float]:
    """Run a strategy‑guided Dijkstra search to get the best path and cost.

    The search begins at `source` and updates the best known cost to each
    node that is connected according to a given `strategy`. The strategy
    controls how edge weights are combined into and how costs are compared
    (e.g. optimising for the shortest‑path or widest‑path).

    If the `target` cannot be reached, the function returns
    `(None, unreached_cost)`, where `unreached_cost` is either `0` or `inf`
    depending on the strategy.

    Otherwise, it returns the optimal path and its cost.

    Args:
        network: PyDiGraph
            The directed graph on which the search is performed.
        source: int
            Index of the starting node.
        target: int
            Index of the destination node.
        strategy: DijkstraStrategy
            Object defining how costs are accumulated and compared.

    Returns:
        path_and_cost:
            A tuple `(path, cost)` where:
            - `path` is a list of node indices forming the optimal route, or
              `None` if the target is unreachable.
            - `cost` is the final cost according to the strategy.
    """
    path: list[int] | None

    if source == target:
        cost = strategy.starting_node_initial_cost
        path = [source]
        return path, cost

    node_is_visited = {n: False for n in network.node_indices()}
    node_best_current_cost = {
        n: strategy.regular_node_unreached_cost for n in network.node_indices()
    }

    node_best_current_cost[source] = strategy.starting_node_initial_cost
    path_map = {source: source}

    candidate_nodes = {source}
    while candidate_nodes:

        def compare_nodes(node1, node2):
            """Wrapper to make _compare_nodes compatible with `cmp_to_key`.

            `cmp_to_key` expects two arguments.
            """
            return _compare_nodes(
                node1, node2, node_best_current_cost, strategy
            )

        order_of_goodness = sorted(
            candidate_nodes, key=cmp_to_key(compare_nodes)
        )
        current_node = order_of_goodness[-1]
        node_is_visited[current_node] = True
        if node_is_visited[target]:
            break

        successors = [
            n
            for n in network.successor_indices(current_node)
            if not node_is_visited[n]
        ]

        for s in successors:
            weight_s = network.get_edge_data(current_node, s)
            proposed_cost = strategy.cost_to(
                node_best_current_cost[current_node], weight_s
            )
            if strategy.is_better_cost(
                proposed_cost, node_best_current_cost[s]
            ):
                node_best_current_cost[s] = proposed_cost
                path_map[s] = current_node

        unvisited_nodes = {
            n for n, is_visited in node_is_visited.items() if not is_visited
        }

        reachable_nodes = {
            n
            for n, best_current_cost in node_best_current_cost.items()
            if best_current_cost != strategy.regular_node_unreached_cost
        }
        candidate_nodes = unvisited_nodes.intersection(reachable_nodes)

    cost = node_best_current_cost[target]
    if cost == strategy.regular_node_unreached_cost:
        path = None
    else:
        path = reconstruct_path(path_map, source, target)

    return path, cost
