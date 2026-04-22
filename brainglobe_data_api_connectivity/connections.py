import csv
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

from rustworkx import PyDiGraph

from ._types import EdgeTable


class Connections:
    """
    A Neurome-browser friendly way of wrapping around a directed graph.

    `rustworkx.PyDiGraph` defines `__new__` which makes inheritance a pain,
    ergo, we shall have an attribute instead.
    """

    _nodes: tuple[NamedTuple]
    network: PyDiGraph

    @classmethod
    def from_files(
        cls,
        nodes: Path,
        connections: Path,
    ) -> "Connections":
        """"""
        # Predict number of edges and nodes that the graph will have
        with nodes.open("r") as node_file:
            node_reader = csv.reader(node_file, delimiter=",")

            # Create Node object with metadata by reading header information
            Node = namedtuple("Node", next(node_reader))

            # Create Node objects themselves
            node_collection = tuple(Node(*row) for row in node_reader)

        with connections.open("r") as edge_file:
            edge_reader = csv.reader(edge_file, delimiter=",")

            # Currently no filter on weight 0, so will add an edge
            edge_table: EdgeTable = tuple(
                (int(row[0]), int(row[1]), float(row[2]))
                for row in edge_reader
            )

        return cls(edge_table=edge_table, nodes=node_collection)

    @property
    def node_class(self) -> type[NamedTuple]:
        """Returns the class of this network's nodes."""
        return self.network.get_node_data(0).__class__

    @property
    def node_fields(self) -> tuple[str]:
        """Return the names of the metadata fields for nodes stored in this
        network.
        """
        return self.network.get_node_data(0)._fields

    def __init__(
        self,
        edge_table: EdgeTable,
        nodes: tuple[NamedTuple],
        name: str = "connections",
    ):
        self.network = PyDiGraph(
            check_cycle=False,
            multigraph=False,
            attrs={"name": name},
            node_count_hint=len(nodes),
            edge_count_hint=len(edge_table),
        )
        # Fairly sure this should always be sequential integers...
        # but better safe than sorry
        _internal_node_indexes = self.network.add_nodes_from(nodes)
        self.network.add_edges_from(
            [
                (
                    _internal_node_indexes[row[0]],
                    _internal_node_indexes[row[1]],
                    row[2],
                )
                for row in edge_table
            ]
        )
        # If we are storing information about non-connections (or connections
        # in general), we also need to do the same sanitisation with
        # _internal_node_indexes on that info too

    def __getattr__(self, name: str):
        """Fallback fast-access of graph attributes."""
        return self.network.attrs.get(name)
