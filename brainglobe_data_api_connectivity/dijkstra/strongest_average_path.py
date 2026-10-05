from rustworkx import PyDiGraph


def strongest_average_path(
    network: PyDiGraph, source: int, target: int
) -> tuple[list[int] | None, float]:
    """Find the route with the highest average connection strength.

    A route follows directed connections from the source region to the target
    region. Its strength is the arithmetic mean of its edge weights:
    total connection strength divided by the number of connections.
    Larger weights are treated as stronger connections.

    Each region may appear only once in a route. This prevents a route from
    repeatedly following a strong cycle to increase its average. Every such
    route is checked, so the result is exact. This can be slow for large or
    densely connected networks with many possible routes.

    Routes are explored one at a time using depth-first search. Different
    routes to the same region are kept because a higher average on the way
    there does not necessarily give a higher average at the final target.

    Args:
        network: PyDiGraph
            Directed network with finite numeric edge weights and at most one
            edge per ordered pair of regions, as used by `Connections.network`.
        source: int
            Internal index of the starting region.
        target: int
            Internal index of the destination region.

    Returns:
        path_and_strength:
            Tuple containing the route's internal node indexes and its average
            edge weight. If no route exists, returns `(None, -inf)`.
            For identical source and target, returns `([source], 0.0)` by
            convention, since that route has no connections to average.
            If several routes share the highest average, returns the first
            one found; no particular order is guaranteed.

    Raises:
        ValueError:
            If either region's internal index does not exist in the network.

    Example:
        Three regions have these directed connections:

            A -> B: strength 8
            B -> C: strength 6
            A -> C: strength 5

        The route A -> B -> C has average strength (8 + 6) / 2 = 7,
        which is stronger on average than the direct route A -> C (5).

        >>> network = PyDiGraph(multigraph=False)
        >>> _ = network.add_nodes_from(["A", "B", "C"])
        >>> _ = network.add_edges_from([(0, 1, 8), (1, 2, 6), (0, 2, 5)])
        >>> strongest_average_path(network, 0, 2)
        ([0, 1, 2], 7.0)

        For a `Connections` instance, pass `connections.network` and the
        internal indexes of the two regions.
    """
    if not network.has_node(source) or not network.has_node(target):
        raise ValueError("Source and target must exist in the network")
    if source == target:
        return [source], 0.0

    best_path = None
    best_average = float("-inf")
    routes_to_explore = [([source], 0.0)]

    while routes_to_explore:
        path, total_strength = routes_to_explore.pop()
        current_region = path[-1]

        if current_region == target:
            average_strength = total_strength / (len(path) - 1)
            if average_strength > best_average:
                best_path = path
                best_average = average_strength
            continue

        for next_region in network.successor_indices(current_region):
            if next_region in path:
                continue
            strength = network.get_edge_data(current_region, next_region)
            routes_to_explore.append(
                (path + [next_region], total_strength + strength)
            )

    return best_path, best_average
