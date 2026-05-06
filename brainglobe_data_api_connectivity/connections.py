from pathlib import Path
from typing import Hashable

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
                    - `edge_meta_to_col`,
                    - `nodes_already_indexed_by`.

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
        nodes_already_indexed_by: str | None = None,
    ):
        """Create a new set of connections.

        Args:
            edge_table: EdgeTable
                Edge-table representation of the node connections; a container
                    of `[from, to, weight]` values. `from` and `to` values
                    should refer to nodes by their identifier in `nodes`.
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
            nodes_already_indexed_by: str | None
                Column header in `nodes` that is being used as the unique
                    identifier (index) for the nodes in the `edge_table` and
                    `edge_meta` inputs. If not provided, the method assumes the
                    row index of a node in `nodes` is being used as the
                    identifier.
        """
        index_translations = self._setup_network(
            nodes, edge_table, existing_node_indexing=nodes_already_indexed_by
        )

        self._setup_edge_metadata(
            edge_meta,
            edge_meta_from_col,
            edge_meta_to_col,
            index_translations=index_translations,
        )

    def _setup_network(
        self,
        nodes: pl.DataFrame,
        edge_table: EdgeTable,
        existing_node_indexing: str | None = None,
    ) -> dict[Hashable, int] | None:
        """Creates the underlying network representation of the connections.

        Sets up the `network` attribute by initialising the underlying graph
        object, and adding the nodes and edges that belong to the graph to this
        object.

        The `nodes` attribute is also set during this method. This is a table
        where table index `i` contains any information about the node with
        (internal) index `i` in the underlying network object.

        At the end of this method, `self.network` and `self.nodes` are set.
        """
        n_nodes = len(nodes)

        if self._node_internal_index_col in nodes:
            raise ValueError(
                f"Heading '{self._node_internal_index_col}' must not be "
                "present in the node metadata, as it is reserved "
                "for internal index referencing."
            )
        if existing_node_indexing is not None:
            if existing_node_indexing not in nodes:
                raise ValueError(
                    f"Heading {existing_node_indexing} "
                    "not present in node metadata."
                )
            n_unique_entries = len(
                nodes.get_column(existing_node_indexing).unique()
            )
            if n_unique_entries != n_nodes:
                raise ValueError(
                    "Given node index column contains repeat entries,"
                    " thus cannot be used as an index."
                )

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
            **{
                self._node_internal_index_col: pl.Series(
                    [i for i in assigned_indexes]
                )
            }
        )

        if existing_node_indexing is None:
            self.network.add_edges_from(edge_table)
            index_translations = None
        else:
            # Nodes are already indexed by a column in the node metadata,
            # so the internal indexes assigned to them do not necessarily match
            # their references in the edge table (and edge metadata either).
            # Adapt as appropriate.
            index_translations = {
                row_dict[existing_node_indexing]: row_dict[
                    self._node_internal_index_col
                ]
                for row_dict in self.nodes.iter_rows(named=True)
            }
            updated_edge_table = [
                (
                    index_translations[old_from],
                    index_translations[old_to],
                    weight,
                )
                for old_from, old_to, weight in edge_table
            ]
            self.network.add_edges_from(updated_edge_table)
        return index_translations

    def _setup_edge_metadata(
        self,
        edge_meta: pl.DataFrame | None,
        from_column: str,
        to_column: str,
        index_translations: dict[Hashable, int] | None = None,
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

            self.ei_from_col = from_column
            self.ei_to_col = to_column

            # Apply any index translations that occurred due to nodes not being
            # indexed by row when they were read in.
            if index_translations is not None:
                self.edge_info = edge_meta.with_columns(
                    pl.col(self.ei_from_col)
                    .replace(index_translations)
                    .alias(self.ei_from_col),
                    pl.col(self.ei_to_col)
                    .replace(index_translations)
                    .alias(self.ei_to_col),
                )
            else:
                self.edge_info = edge_meta
        else:
            self.edge_info = None
            self.ei_from_col = None
            self.ei_to_col = None
