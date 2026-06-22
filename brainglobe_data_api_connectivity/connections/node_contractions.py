"""Weight-contraction functions for use in `Connections.contract_nodes`.

`rustworkx` passes edge weights (of groups of edges that are to be combined)
into a function that must take a variable number of positional arguments only.
As such, several of the functions here are just convenience wrappers around
existing Python functions that want to take `Iterable`s instead of a number of
variable positional arguments.
"""


def sum_of_weights(*args: float) -> float:
    """Combined edge weight is the sum of all constituent edge weights."""
    return sum(args)


def min_of_weights(*args: float) -> float:
    """Combined edge weight is the minimum of all constituent edge weights."""
    return min(args)


def max_of_weights(*args: float) -> float:
    """Combined edge weight is the maximum of all constituent edge weights."""
    return max(args)


def average_of_weights(*args: float) -> float:
    """Combined edge weight is the average of all constituent edge weights."""
    return sum(args) / len(args)
