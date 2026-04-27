import csv
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from rustworkx import PyDiGraph

from ._types import EdgeTable


class Connections:
    """
    A Neurome-browser friendly way of wrapping around a directed graph.

    `rustworkx.PyDiGraph` defines `__new__` which makes inheritance a pain,
    ergo, we shall have an attribute instead.
    """

    edge_info: pd.DataFrame | None
    network: PyDiGraph

    @classmethod
    def from_files(
        cls,
        nodes: Path,
        edge_table: Path,
        edge_meta: Path | None = None,
        **constructor_kwargs,
    ) -> "Connections":
        """"""
        with nodes.open("r") as node_file:
            node_reader = csv.reader(node_file, delimiter=",")

            # TO CONSIDER: Possibly better for us to just use node
            # indexes, and use something like a dataframe to allow access
            # to information?

            # Create Node object with metadata by reading header information
            header_row = next(node_reader)
            Node = namedtuple("Node", header_row)

            # Create Node objects themselves
            node_collection = tuple(Node(*row) for row in node_reader)

        with edge_table.open("r") as edge_file:
            edge_reader = csv.reader(edge_file, delimiter=",")

            # Currently no filter on weight 0, so will add an edge!
            edge_table_entries = tuple(
                (int(row[0]), int(row[1]), float(row[2]))
                for row in edge_reader
            )

        if edge_meta is not None:
            edge_meta = pd.read_csv(edge_meta, delimiter=",")

        return cls(
            edge_table=edge_table_entries,
            nodes=node_collection,
            edge_meta=edge_meta,
            **constructor_kwargs,
        )

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
        edge_meta: pd.DataFrame | None = None,
        *,
        edge_meta_node_from_col: str = "from",
        edge_meta_node_to_col: str = "to",
    ):
        self.network = PyDiGraph(
            check_cycle=False,
            multigraph=False,
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

        # If we are also storing edge metadata, we need to update the
        # "from" and "to" node reference columns to the internal node
        # indexes as well.
        if edge_meta is not None:
            if edge_meta_node_from_col == edge_meta_node_to_col:
                raise ValueError(
                    "Connection metadata 'from' and 'to' columns are the same!"
                )

            edge_meta[edge_meta_node_from_col].map(
                lambda x: _internal_node_indexes[x]
            )
            edge_meta[edge_meta_node_to_col].map(
                lambda x: _internal_node_indexes[x]
            )

            edge_meta.set_index(
                [edge_meta_node_from_col, edge_meta_node_to_col],
                drop=True,
                inplace=True,
            )
        self.edge_info = edge_meta
