from pathlib import Path

import polars as pl
from rustworkx import PyDiGraph

from ._types import EdgeTable


class Connections:
    """
    A Neurome-browser friendly way of wrapping around a directed graph.

    `rustworkx.PyDiGraph` defines `__new__` which makes inheritance a pain,
    ergo, we shall have an attribute instead.
    """

    _edge_meta_index_col: str = "graph_edge"
    _node_internal_index_col: str = "node_index"

    edge_info: pl.DataFrame | None
    ei_from_col: str | None
    ei_to_col: str | None

    network: PyDiGraph
    nodes: pl.DataFrame

    @classmethod
    def from_files(
        cls,
        nodes: Path,
        edge_table: Path,
        edge_meta: Path | None = None,
        **constructor_kwargs,
    ) -> "Connections":
        """Create `Connections` by reading information from a file.

        TODO: Update me if we decide to include an info.json file inside a
        folder, in which case the path to the info.json file (or the containing
        folder) should be the input(s).

        NOTE: Like the constructor method, this method assumes that the nodes
        are referred to by their row-index in the `nodes` file in both the edge
        table and any edge metadata.

        Args:
            nodes: Path
                Path to the file containing information about regions (nodes).
            edge_table: Path
                Path to the file containing the edge table, to read information
                    about connected regions.
            edge_meta: Path
                Path to the file containing metadata about edge connections.
            constructor_kwargs:
                Additional keyword arguments to pass to the constructor method.
                    Accepts
                    - `edge_meta_from_col`,
                    - `edge_meta_to_col`.

                    See `Connections.__init__` for more information.

        Returns:
            connections: `Connections` instance created from file information.
        """
        node_collection = pl.read_csv(nodes)

        edge_table_entries = list(
            pl.read_csv(edge_table, has_header=False).iter_rows(named=False)
        )

        if edge_meta is not None:
            edge_meta = pl.read_csv(edge_meta)

        return cls(
            edge_table=edge_table_entries,
            nodes=node_collection,
            edge_meta=edge_meta,
            **constructor_kwargs,
        )

    def __init__(
        self,
        edge_table: EdgeTable,
        nodes: pl.DataFrame,
        edge_meta: pl.DataFrame | None = None,
        *,
        edge_meta_from_col: str = "from",
        edge_meta_to_col: str = "to",
    ):
        """Create a new set of connections.

        Args:
            edge_table: EdgeTable
                Edge-table representation of the node connections; a container
                    of `[from, to, weight]` values. `from` and `to` values
                    should refer to nodes by the index of their corresponding
                    row in `nodes`.
            nodes: pd.DataFrame
                DataFrame containing information about the nodes. The index
                    will be overwritten by the internal indexes used for the
                    nodes by the internal network representation.
            edge_meta: pd.DataFrame
                DataFrame containing information about connections. Two columns
                    must be present that contain the index (of the
                    corresponding row in `nodes`) of the node "from" which the
                    edge leaves and "to" which the edge connects. All other
                    columns are assumed to contain data.
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
        nodes: pl.DataFrame,
        edge_table: EdgeTable,
    ) -> None:
        """Creates the underlying network representation of the connections.

        Sets up the `network` attribute by initialising the underlying graph
        object, and adding the nodes and edges that belong to the graph to this
        object.

        The `nodes` attribute is also set during this method. This is a table
        where table index `i` contains any information about the node with
        (internal) index `i` in the underlying network object.

        At the end of this method, `self.network` and `self.nodes` are set.
        """
        if self._node_internal_index_col in nodes:
            raise ValueError(
                f"Heading '{self._node_internal_index_col}' must not be "
                "present in the node metadata, as it is reserved"
                "for internal index referencing."
            )

        n_nodes = len(nodes)

        self.network = PyDiGraph(
            check_cycle=False,
            multigraph=False,
            node_count_hint=n_nodes,
            edge_count_hint=len(edge_table),
        )

        # Internal node indexes should just be the range from 0 -> n_nodes-1,
        # since rustworkx graphs assign sequential indexes to given nodes.
        assigned_indexes = self.network.add_nodes_from(range(n_nodes))

        self.nodes = nodes.with_columns(
            **{self._node_internal_index_col: [i for i in assigned_indexes]}
        )
        self.network.add_edges_from(edge_table)

    def _setup_edge_metadata(
        self,
        edge_meta: pl.DataFrame | None,
        from_column: str,
        to_column: str,
    ) -> None:
        """Setup storage for edge (connection) metadata.

        Note that edge metadata is optional, and the corresponding attribute is
        set to `None` if no such information is provided.

        At the end of this method, `self.edge_info` is set, as well as both of
        `self.ei_{from,to}_col`.
        """
        if edge_meta is not None:
            if from_column == to_column:
                raise ValueError(
                    "Connection metadata 'from' and 'to' columns are the same "
                    f"({from_column})."
                )

            self.edge_info = edge_meta
            self.ei_from_col = from_column
            self.ei_to_col = to_column
        else:
            self.edge_info = None
            self.ei_from_col = None
            self.ei_to_col = None
