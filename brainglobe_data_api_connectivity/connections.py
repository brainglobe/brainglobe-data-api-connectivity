import csv
from pathlib import Path

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
    nodes: pd.DataFrame

    @classmethod
    def from_files(
        cls,
        nodes: Path,
        edge_table: Path,
        edge_meta: Path | None = None,
        node_index_column: str | None = None,
        **constructor_kwargs,
    ) -> "Connections":
        """"""
        node_collection = pd.read_csv(
            nodes, delimiter=",", index_col=node_index_column
        )

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

    def __init__(
        self,
        edge_table: EdgeTable,
        nodes: pd.DataFrame,
        edge_meta: pd.DataFrame | None = None,
        *,
        edge_meta_from_col: str = "from",
        edge_meta_to_col: str = "to",
    ):
        """Create a new set of connections.

        Args:
            edge_table: EdgeTable
                Edge-table representation of the node connections; a container
                    of `[from, to, weight]` values.
            nodes: pd.DataFrame
                DataFrame containing information about the nodes. The index
                    will be overwritten by the internal indexes used for the
                    nodes by the internal network representation.
            edge_meta: pd.DataFrame
                DataFrame containing information about connections. Two columns
                    must be present that contain the index of the node "from"
                    which the edge leaves and "to" which the edge connects. All
                    other columns are assumed to contain data.
            edge_meta_from_col: str
                Header of the column in the `edge_meta` argument containing the
                    "from" node indexes.
            edge_meta_to_col: str
                Header of the column in the `edge_meta` argument containing the
                    "to" node indexes.
        """
        self._setup_network(nodes, edge_table)

        self._setup_edge_metadata(
            edge_meta, edge_meta_from_col, edge_meta_to_col
        )

    def _setup_network(
        self,
        nodes: pd.DataFrame,
        edge_table: EdgeTable,
    ) -> None:
        """Creates the underlying network representation of the connections.

        Sets up the `network` attribute by initialising the underlying graph
        object, and adding the nodes and edges that belong to the graph to this
        object.

        The `nodes` attribute is also set during this method. This is a table
        where table index `i` contains any information about the node with
        index `i` in the underlying network object.

        Note that the underlying network object chooses its own internal
        indexing for the nodes (and thus edges). To ensure consistency, we
        first create the nodes in the graph from the index of the table
        provided, then swap out this indexing method for the indexes assigned
        to each node by the graph.

        At the end of this method, `self.network` and `self.nodes` are set.
        """
        self.network = PyDiGraph(
            check_cycle=False,
            multigraph=False,
            node_count_hint=len(nodes),
            edge_count_hint=len(edge_table),
        )

        # Fairly sure this should always be sequential integers...
        # but better safe than sorry
        self.nodes = nodes
        _internal_node_indexes = self.network.add_nodes_from(self.nodes.index)
        # self.nodes.index.map(lambda x: _internal_node_indexes[x])
        self.nodes.rename(
            index={
                old_index: _internal_node_indexes[i]
                for i, old_index in enumerate(self.nodes.index)
            },
            inplace=True,
        )

        self.network.add_edges_from(
            [
                (
                    self.nodes.index[row[0]],
                    self.nodes.index[row[1]],
                    row[2],
                )
                for row in edge_table
            ]
        )

    def _setup_edge_metadata(
        self,
        edge_meta: pd.DataFrame | None,
        from_column: str,
        to_column: str,
    ) -> None:
        """Setup storage for edge (connection) metadata.

        Edge metadata is stored as a `DataFrame`. Rows are indexed using a
        `MultiIndex` of the form `(i, j)` to represent the edge from `i` to
        `j`. This allows lookup of connection information via the `.loc[i, j]`.

        Note that edge metadata is optional, and the corresponding attribute is
        set to `None` if no such information is provided.

        At the end of this method, `self.edge_info` is set.
        """
        self.edge_info = edge_meta

        if self.edge_info is not None:
            if from_column == to_column:
                raise ValueError(
                    "Connection metadata 'from' and 'to' columns are the same "
                    f"({from_column})."
                )

            # If we are also storing edge metadata, we need to update the
            # "from" and "to" node reference columns to use the internal node
            # indexes as well.
            self.edge_info[from_column].map(lambda x: self.nodes.index[x])
            self.edge_info[to_column].map(lambda x: self.nodes.index[x])

            self.edge_info.set_index(
                [from_column, to_column],
                drop=True,
                inplace=True,
            )
