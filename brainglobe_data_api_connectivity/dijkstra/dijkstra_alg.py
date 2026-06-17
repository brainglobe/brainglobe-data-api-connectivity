from functools import cmp_to_key

from rustworkx import PyDiGraph

from .strategy import DijkstraStrategy


def _foo(node1, node2, node_best_current_cost, strategy: DijkstraStrategy):
    cost_node1 = node_best_current_cost[node1]
    cost_node2 = node_best_current_cost[node2]
    return 1 if strategy.is_better_cost(cost_node1, cost_node2) else -1


def reconstruct_path(
    previous_node: dict[int, int], starting_node: int, destination_node: int
) -> list[int]:
    path_list = [destination_node]
    current_node = destination_node
    while previous_node[current_node] != starting_node:
        current_node = previous_node[current_node]
        path_list.append(current_node)
    path_list.append(starting_node)
    path_list.reverse()
    return path_list


def dijkstra(
    network: PyDiGraph, source: int, target: int, strategy: DijkstraStrategy
) -> tuple[list[int] | None, float]:
    """"""
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
    previous_node = {source: source}

    candidate_nodes = {source}
    while candidate_nodes:

        def foo(node1, node2):
            return _foo(node1, node2, node_best_current_cost, strategy)

        order_of_goodness = sorted(candidate_nodes, key=cmp_to_key(foo))
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
                previous_node[s] = current_node

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
        path = reconstruct_path(previous_node, source, target)

    return path, cost
