from pathlib import Path
from typing import Container, Hashable

import polars as pl
from rustworkx import PyDiGraph

from ._types import EdgeTable


class Connections:
    """
    A Neurome-browser friendly way of wrapping around a directed graph.

    `rustworkx.PyDiGraph` defines `__new__` which makes inheritance a pain,
    ergo, we shall have an attribute instead.
    """

    _node_internal_index_col: str = "node_index"

    edge_info: pl.DataFrame | None
    edge_info_from_col: str | None
    edge_info_to_col: str | None

    network: PyDiGraph
    nodes: pl.DataFrame

    @classmethod
    def from_files(
        cls,
        node_info: Path,
        edge_table: Path,
        edge_info: Path | None = None,
        **constructor_kwargs,
    ) -> "Connections":
        """Create `Connections` by reading information from a file.

        TODO: Update me if we decide to include an info.json file inside a
        folder, in which case the path to the info.json file (or the containing
        folder) should be the input(s).

        Args:
            node_info: Path
                Path to the file containing information about regions (nodes).
            edge_table: Path
                Path to the file containing the edge table, to read information
                    about connected regions.
            edge_info: Path
                Path to the file containing metadata about edge connections.
            constructor_kwargs:
                Additional keyword arguments to pass to the constructor method.
                    Accepts
                    - `edge_info_from_col`,
                    - `edge_info_to_col`,
                    - `node_index_column`.

                    See `Connections.__init__` for more information.

        Returns:
            connections: `Connections` instance created from file information.
        """
        node_collection = pl.read_csv(node_info)

        edge_table_entries = list(
            pl.read_csv(edge_table, has_header=False).iter_rows(named=False)
        )

        if edge_info is not None:
            edge_info = pl.read_csv(edge_info)

        return cls(
            node_info=node_collection,
            edge_table=edge_table_entries,
            edge_info=edge_info,
            **constructor_kwargs,
        )

    def __init__(
        self,
        node_info: pl.DataFrame,
        edge_table: EdgeTable,
        edge_info: pl.DataFrame | None = None,
        *,
        edge_info_from_col: str = "from",
        edge_info_to_col: str = "to",
        node_index_column: str | None = None,
    ):
        """Create a new set of connections.

        Args:
            node_info: pl.DataFrame
                DataFrame containing information about the nodes. The index
                    will be overwritten by the internal indexes used for the
                    nodes by the internal network representation.
            edge_table: EdgeTable
                Edge-table representation of the node connections; a container
                    of `[from, to, weight]` values. `from` and `to` values
                    should refer to nodes by their identifier in `nodes`.
            edge_info: pl.DataFrame
                DataFrame containing information about connections. Two columns
                    must be present that contain the index (of the
                    corresponding row in `nodes`) of the node "from" which the
                    edge leaves and "to" which the edge connects. All other
                    columns are assumed to contain data.
            edge_info_from_col: str
                Header of the column in the `edge_info` argument containing the
                    "from" node indexes.
            edge_info_to_col: str
                Header of the column in the `edge_info` argument containing the
                    "to" node indexes.
            node_index_column: str | None
                Column header in `nodes` that is being used as the unique
                    identifier (index) for the nodes in the `edge_table` and
                    `edge_info` inputs. If not provided, the method assumes the
                    row index of a node in `nodes` is being used as the
                    identifier.
        """
        index_translations = self._setup_network(
            node_info, edge_table, existing_node_indexing=node_index_column
        )

        self._setup_edge_metadata(
            edge_info,
            edge_info_from_col,
            edge_info_to_col,
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
        if self._node_internal_index_col in nodes:
            raise ValueError(
                f"Heading '{self._node_internal_index_col}' must not be "
                "present in the node metadata, as it is reserved "
                "for internal index referencing."
            )

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
        edge_info: pl.DataFrame | None,
        from_column: str,
        to_column: str,
        index_translations: dict[Hashable, int] | None = None,
    ) -> None:
        """Setup storage for edge (connection) metadata.

        Note that edge metadata is optional, and the corresponding attribute is
        set to `None` if no such information is provided.

        At the end of this method, `self.edge_info` is set, as well as both of
        `self.edge_info_{from,to}_col`.
        """
        if edge_info is not None:
            if from_column == to_column:
                raise ValueError(
                    "Connection metadata 'from' and 'to' columns are the same "
                    f"({from_column})."
                )

            self.edge_info_from_col = from_column
            self.edge_info_to_col = to_column

            # Apply any index translations that occurred due to nodes not being
            # indexed by row when they were read in.
            if index_translations is not None:
                self.edge_info = edge_info.with_columns(
                    pl.col(self.edge_info_from_col)
                    .replace(index_translations)
                    .alias(self.edge_info_from_col),
                    pl.col(self.edge_info_to_col)
                    .replace(index_translations)
                    .alias(self.edge_info_to_col),
                )
            else:
                self.edge_info = edge_info
        else:
            self.edge_info = None
            self.edge_info_from_col = None
            self.edge_info_to_col = None

    def _node_indexes_from_information(
        self, *predicates, **constraints
    ) -> pl.Series:
        """Return graph indexes of nodes that match the given information.

        This is essentially a convenience wrapper around a `DataFrame` `filter`
        followed by a `get_column`. Intended use is so that users can select
        nodes by neurological (?) information, and the API then handles
        translating this information into the relevant internal node indexes,
        running the actual graph-theoretic query, and then returning
        the results.

        Graph indexes are stored in the `.nodes` attribute, in the
        `self._node_internal_index_col` column. The values in this column,
        whose other row values match the given filters, are returned by this
        function.

        Args:
            predicates:
                See [`polars.DataFrame.filter`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.filter.html).
            constraints:
                See [`polars.DataFrame.filter`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.filter.html).

        Returns:
            graph_indexes:
                Series containing all internal node indexes in `self.network`
                that correspond to nodes with the given metadata.
        """
        return self.nodes.filter(*predicates, **constraints).get_column(
            self._node_internal_index_col
        )

    def _node_information_from_index(
        self, node_indexes: Container[int]
    ) -> pl.DataFrame:
        """Return information about nodes with the selected (internal) indexes.

        Essentially a wrapper around a `polars.DataFrame.filter` that looks up
        the relevant rows from the `.nodes` attribute and returns them.

        Args:
            node_indexes: Container[int]
                Internal node indexes, referencing nodes to fetch information
                about.

        Returns:
            node_information:
                `polars.DataFrame` whose rows contain node information for the
                requested nodes.
        """
        return self.nodes.filter(
            pl.col(self._node_internal_index_col).is_in(node_indexes)
        )
