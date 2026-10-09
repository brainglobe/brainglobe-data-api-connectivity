from rustworkx import PyDiGraph


def strongest_average_path(
    network: PyDiGraph,
    source: int,
    target: int,
    max_steps: int = 5,
) -> tuple[list[int] | None, float]:
    """Find the route with the highest average connection strength.

    A route follows directed connections from the source region to the target
    region. Its strength is the arithmetic mean of its edge weights:
    total connection strength divided by the number of connections.
    Larger weights are treated as stronger connections.

    Each region may appear only once in a route, except that the starting
    region may also be the final region when searching for feedback cycles
    (source == target). No other repeated regions are allowed.

    All routes within `max_steps` are considered, ensuring an exact result.
    The default limit is 5. Execution time can increase significantly with
    larger limits, especially in dense networks.

    Routes are explored one at a time using depth-first search.

    To speed up the search, routes that cannot improve the best result found
    so far are skipped. For each partial route, the algorithm calculates the
    highest average it could theoretically achieve by assuming every remaining
    connection has the strongest edge weight in the entire network. The actual
    average of any completed route cannot be higher than this estimate, so
    there is no reason to explore that route further.

    If there is a direct connection to the target, this is also checked before
    the search starts, providing an initial best result that can help skip
    unpromising routes.

    These optimizations reduce the number of routes explored without
    excluding any route that could have a higher average connection strength.
    The result therefore remains exact within `max_steps`.

    Args:
        network: PyDiGraph
            Directed network with finite numeric edge weights and at most one
            edge per ordered pair of regions, as used by `Connections.network`.
        source: int
            Internal index of the starting region.
        target: int
            Internal index of the destination region.
        max_steps: int | None
            Maximum number of edges in a route. None searches all simple
            routes, including feedback cycles. A limit can exclude a stronger
            route with more steps.

    Returns:
        path_and_strength: tuple[list[int] | None, float]
            Tuple containing the route's internal node indexes and its average
            edge weight. If no route exists within the limit, returns
            `(None, -inf)`.
            For identical source and target, searches for the strongest
            nonempty feedback cycle, including self-loops.
            If several routes share the highest average, returns the first
            one found; no particular order is guaranteed.

    Raises:
        ValueError:
            If either region's internal index does not exist in the network.
            Also raised if `max_steps` is negative.

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
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    best_path = None
    best_average = float("-inf")
    maximum_strength = max(network.edges(), default=float("-inf"))

    # Use a direct connection as the initial best route, if available.
    if (max_steps is None or max_steps > 0) and network.has_edge(
        source, target
    ):
        best_path = [source, target]
        best_average = network.get_edge_data(source, target)

    # Explore routes
    routes_to_explore = [([source], 0.0)]

    while routes_to_explore:
        path, total_strength = routes_to_explore.pop()
        current_region = path[-1]
        steps = len(path) - 1

        # Evaluate completed routes and update the best result.
        if steps > 0 and current_region == target:
            average_strength = total_strength / steps
            if best_path is None or average_strength > best_average:
                best_path = path
                best_average = average_strength
            continue

        if max_steps is not None and steps >= max_steps:
            continue

        # Skip routes that cannot beat the best average.
        if best_path is not None:
            remaining_steps = (
                max_steps - steps
                if max_steps is not None
                else network.num_nodes() - steps
            )
            optimistic_average = max(
                (total_strength + k * maximum_strength) / (steps + k)
                for k in range(1, remaining_steps + 1)
            )
            if optimistic_average <= best_average:
                continue

        # With one step left, only the target can be reached.
        if max_steps is not None and steps == max_steps - 1:
            successors = (
                [target] if network.has_edge(current_region, target) else []
            )
        else:
            successors = network.successor_indices(current_region)

        # No repeated nodes, except when source == target (feedback loop).
        for next_region in successors:
            if next_region in path and not (next_region == source == target):
                continue

            strength = network.get_edge_data(current_region, next_region)
            routes_to_explore.append(
                (path + [next_region], total_strength + strength)
            )

    return best_path, best_average
