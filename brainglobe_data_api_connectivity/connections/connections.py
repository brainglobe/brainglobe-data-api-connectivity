import warnings
from pathlib import Path
from typing import Any, Callable, Container, Hashable, Iterable

import polars as pl
from rustworkx import PyDiGraph

from .._types import EdgeTable
from .node_contractions import sum_of_weights
from .query_opts import ConnectionsLookup, NodeIs


class Connections:
    """
    A Neurome-browser friendly way of wrapping around a directed graph.

    `rustworkx.PyDiGraph` defines `__new__` which makes inheritance a pain,
    ergo, we shall have an attribute instead.
    """

    _node_internal_index_col: str = "__node_index"
    _edge_info_from_index_col: str = "__idx_from"
    _edge_info_to_index_col: str = "__idx_to"

    _user_node_index_col: str | None = None

    edge_info: pl.DataFrame | None
    edge_info_from_col: str | None
    edge_info_to_col: str | None

    network: PyDiGraph
    nodes: pl.DataFrame
    collapsed_node_indexes: set[int]

    @property
    def node_index_column(self) -> str | None:
        """Return the column in `self.node_info` or unique identifiers.

        If the user provided a `node_index_column` at creation, the name of
        this column is returned. Otherwise, the instance will have assigned
        internal indexes assuming the indexing of the rows in `node_info` was
        being used as the reference value in the `edge_table` and `edge_info`,
        in which case `None` is returned.
        """
        return self._user_node_index_col

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
        self.collapsed_node_indexes = set()

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

        At the end of this method, `self.network`, `self.nodes`, and
        `self._user_node_index_col` are set.
        """
        if self._node_internal_index_col in nodes:
            raise ValueError(
                f"Heading '{self._node_internal_index_col}' must not be "
                "present in the node metadata, as it is reserved "
                "for internal index referencing."
            )
        self._user_node_index_col = existing_node_indexing

        n_nodes = len(nodes)

        if self._user_node_index_col is not None:
            if self._user_node_index_col not in nodes:
                raise ValueError(
                    f"Heading {self._user_node_index_col} "
                    "not present in node metadata."
                )
            n_unique_entries = len(
                nodes.get_column(self._user_node_index_col).unique()
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

        if self._user_node_index_col is None:
            self.network.add_edges_from(edge_table)
            index_translations = None
        else:
            # Nodes are already indexed by a column in the node metadata,
            # so the internal indexes assigned to them do not necessarily match
            # their references in the edge table (and edge metadata either).
            # Adapt as appropriate.
            index_translations = {
                row_dict[self._user_node_index_col]: row_dict[
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
            for reserved_header in [
                self._edge_info_from_index_col,
                self._edge_info_to_index_col,
            ]:
                if reserved_header in edge_info.columns:
                    raise ValueError(
                        f"Heading '{reserved_header}' must not be "
                        "present in the edge metadata, as it is reserved "
                        "for internal index referencing."
                    )

            self.edge_info_from_col = from_column
            self.edge_info_to_col = to_column

            # Apply any index translations that occurred due to nodes not being
            # indexed by row when they were read in.
            if index_translations is not None:
                self.edge_info = edge_info.with_columns(
                    pl.col(self.edge_info_from_col)
                    .replace(index_translations)
                    .alias(self._edge_info_from_index_col),
                    pl.col(self.edge_info_to_col)
                    .replace(index_translations)
                    .alias(self._edge_info_to_index_col),
                )
            else:
                self.edge_info = edge_info.with_columns(
                    pl.col(self.edge_info_from_col).alias(
                        self._edge_info_from_index_col
                    ),
                    pl.col(self.edge_info_to_col).alias(
                        self._edge_info_to_index_col
                    ),
                )
        else:
            self.edge_info = None
            self.edge_info_from_col = None
            self.edge_info_to_col = None

    def contract_nodes(
        self,
        nodes: Iterable[int],
        weight_contraction_fn: Callable[..., float] = sum_of_weights,
        super_node_info: dict[str, Any] | None = None,
    ) -> int:
        r"""Collapse nodes into a single node.

        This operation is done **in-place**, modifying the `.network`. Indices
        passed in `nodes` will no longer be present in the `.network`, and
        a new node that represents the "super"-node will be added. The
        index of this node is returned by the method.

        Note that any metadata pertaining to the contracted nodes, and the
        resulting region, is preserved. The nodes that are "contracted" will
        have their internal indexes added to `collapsed_node_indexes`, to
        reflect the fact that they are no longer represented in the network.
        The super-node will be added to the `.nodes` table.

        Collapsing several nodes into a single node removes any edges between
        pairs of said nodes.

        If multiple nodes in `nodes` $v_i$ (for some index set
        $i\in\mathcal{I}$) have a connection to some other node $v_j$ that is
        not in `nodes`, then a decision must be made regarding the weight of
        the resulting connection between the super-node and $v_j$. This is
        based on the weights of the edges $(v_i, v_j)$ (reverse direction is
        treated separately but identically), which are passed as arguments to
        the `weight_contraction_fn` and should return the resulting weight that
        will be applied to the edge from the super-node to $v_j$. Common
        contraction options are available in the `.node_contractions` module.

        Args:
            nodes: Container[int]
                Internal node indexes that are to be contracted.
            weight_contraction_fn: Callable[..., float]
                Function that dictates behaviour for combining edge weights
                when nodes are contracted (see above). Default is to sum edge
                weights.
            super_node_info: dict[str, Any] | None
                Information about the resulting super-node, created by the
                contraction process. Keys in this dictionary should match
                column headers in `.nodes`.

        Returns:
            super_node_index:
                Internal index of the super-node within the `.network`.
        """
        # Perform collapse, delegating to rustworkx and recording collapsed
        # node indexes.
        super_node_data = "Collapse of " + ", ".join(str(i) for i in nodes)
        super_node_index = self.network.contract_nodes(
            nodes, super_node_data, weight_combo_fn=weight_contraction_fn
        )
        self.collapsed_node_indexes = self.collapsed_node_indexes.union(nodes)

        # Add the super-node to `.nodes`. Populate it with any information
        # provided about the new node, if applicable. Note that information
        # provided that is not an existing column header is ignored.
        if super_node_info is None:
            super_node_info = {}
        super_node_info = {
            key: super_node_info.get(key) for key in self.nodes.columns
        }
        super_node_info[self._node_internal_index_col] = super_node_index
        # NOTE: Should consider .vstack() here too, depending on how many times
        # we expect to contract on a single graph.
        self.nodes.extend(pl.DataFrame(super_node_info))

        # Handling anything hierarchical should then be done here.

        return super_node_index

    def node_indexes_from_information(
        self, *predicates, **constraints
    ) -> pl.Series:
        """Return graph indexes of nodes that match the given information.

        Allows for looking up node indexes (as used by the `.network` object)
        via information stored about the brain regions themselves. For example,
        one can use this function to obtain the node indexes of all brain
        regions that match a particular name, are on a particular side of the
        brain, or a combination of these things. Regions are filtered based on
        whether their corresponding row-entry in the `.nodes` `DataFrame`
        matches the criteria given.

        This is essentially a convenience wrapper around a `DataFrame` `filter`
        followed by a `get_column`. Intended use is so that users can select
        nodes by neuroscientific information, and have the API handle
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

    def node_information_from_index(
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

    def _get_unique_node_index(self, node_id: dict[str, str | int]) -> int:
        """Return the internal index for a node.

        Also checks whether node is valid (existent and unique).

        Args:
            node_id:
                Information identifying the node.

        Returns:
            Internal index of the identified node.

        Raises:
            ValueError:
                If `node_id` does not identify exactly one node.
        """
        node_indexes = self.node_indexes_from_information(**node_id).to_list()

        if len(node_indexes) != 1:
            raise ValueError(
                "Expected 1 unique node, "
                f"but got {len(node_indexes)} for {node_id}."
            )

        return node_indexes[0]

    def _get_available_connection_lookup(
        self,
        connections_lookup: ConnectionsLookup,
    ) -> ConnectionsLookup:
        """Set the connection lookup source based on the available information.

        If all connections are requested but edge information is unavailable,
        warn the user and fall back to connections reported by the graph.

        Args:
            connections_lookup:
                Requested source for connection lookup.

        Returns:
            The requested lookup source if available; otherwise,
            `ConnectionsLookup.REPORTED`.
        """
        if (
            connections_lookup == ConnectionsLookup.ALL
            and self.edge_info is None
        ):
            warnings.warn(
                "No edge information available. "
                "Using graph information instead.",
                UserWarning,
            )
            return ConnectionsLookup.REPORTED

        return connections_lookup

    def direct_connections(
        self,
        node_internal_index: int,
        node_as: NodeIs = NodeIs.ANY,
        connections_lookup: ConnectionsLookup = ConnectionsLookup.REPORTED,
    ) -> tuple[list[int], list[int]]:
        """
        Report direct connections of a node.

        When reporting connections, one can choose to use either the `.network`
        or `.edge_info` as the source from which to find connections. This
        choice is handled by the `connections_lookup` option, and should be
        specified using the `ConnectionsLookup` enum.

        By default the method will return all direct connections that the
        node with internal index `node_internal_index` possesses, split into
        two lists by whether this node is the input (source) or output (target)
        in the direct connection. If only one of these lists is desired, the
        `node_as` argument can be passed to specify which, and the method will
        not bother searching for the other. Use the `NodeIs` enum to specify.

        Args:
            node: int
                Index of a node in the network to fetch direct connections of.
            node_as: NodeIs
                The role in the connection that `node` should play, in order to
                be returned.
            connections_lookup: ConnectionsLookup
                The source to use when searching for the `node`s connections.

        Returns:
            connections_as_input:
                List of node indexes to which `node` connects as an input node.
                That is, for each `i` in this list, the edge `(node, i)`
                exists. Will be empty if only output nodes are requested.
            connections_as_output:
                List of node indexes to which `node` connects as an output
                node. That is, for each `i` in this list, the edge `(i, node)`
                exists. Will be empty if only input nodes are requested.

        Raises:
            TypeError:
                When attempting to search `.edge_info` for connection reports,
                but the attribute has not been set.
        """
        connections_as_input = []
        connections_as_output = []

        if connections_lookup == ConnectionsLookup.REPORTED:
            if node_as != NodeIs.OUTPUT:
                connections_as_input = [
                    i
                    for i in self.network.successor_indices(
                        node_internal_index
                    )
                ]
            if node_as != NodeIs.INPUT:
                connections_as_output = [
                    i
                    for i in self.network.predecessor_indices(
                        node_internal_index
                    )
                ]
        else:
            if self.edge_info is None:
                raise TypeError(
                    "Edge information is not assigned "
                    "(`self.edge_info` is `None`)"
                )
            if node_as != NodeIs.OUTPUT:
                connections_as_input = (
                    self.edge_info.filter(
                        pl.col(self._edge_info_from_index_col)
                        == node_internal_index
                    )
                    .get_column(self._edge_info_to_index_col)
                    .unique()
                    .to_list()
                )
            if node_as != NodeIs.INPUT:
                connections_as_output = (
                    self.edge_info.filter(
                        pl.col(self._edge_info_to_index_col)
                        == node_internal_index
                    )
                    .get_column(self._edge_info_from_index_col)
                    .unique()
                    .to_list()
                )

        return connections_as_input, connections_as_output

    def _direct_connection_from_edge_info(
        self,
        node0: int | dict[str, str | int],
        node1: int | dict[str, str | int],
        node0_as: NodeIs,
    ) -> pl.DataFrame:
        """Return direct connections between two nodes from `edge_info`."""

        self.edge_info: pl.DataFrame

        node0_idx, node1_idx = [
            self._get_unique_node_index(node)
            if isinstance(node, dict)
            else node
            for node in [node0, node1]
        ]

        from_col = pl.col(self.edge_info_from_col)
        to_col = pl.col(self.edge_info_to_col)

        if node0_as == NodeIs.INPUT:
            connection_filter = (from_col == node0_idx) & (to_col == node1_idx)

        elif node0_as == NodeIs.OUTPUT:
            connection_filter = (from_col == node1_idx) & (to_col == node0_idx)

        else:
            connection_filter = (
                (from_col == node0_idx) & (to_col == node1_idx)
            ) | ((from_col == node1_idx) & (to_col == node0_idx))

        connections = self.edge_info.filter(connection_filter)
        return pl.DataFrame(connections)

    def _direct_connection_from_network(
        self,
        node0: int | dict[str, str | int],
        node1: int | dict[str, str | int],
        node0_as: NodeIs,
    ) -> pl.DataFrame:
        """Return direct connections between two nodes from the network."""

        node0_idx, node1_idx = [
            self._get_unique_node_index(node)
            if isinstance(node, dict)
            else node
            for node in [node0, node1]
        ]

        node0_id, node1_id = [
            next(iter(node.values())) if isinstance(node, dict) else node
            for node in (node0, node1)
        ]  # stays idx if this is used as input

        connections = []

        if node0_as != NodeIs.OUTPUT:
            if self.network.has_edge(node0_idx, node1_idx):
                edge_data = self.network.get_edge_data(node0_idx, node1_idx)
                connections.append(
                    {"from": node0_id, "to": node1_id, "value": edge_data}
                )

        if node0_as != NodeIs.INPUT:
            if self.network.has_edge(node1_idx, node0_idx):
                edge_data = self.network.get_edge_data(node1_idx, node0_idx)
                connections.append(
                    {"from": node1_id, "to": node0_id, "value": edge_data}
                )

        return pl.DataFrame(connections)

    def direct_connection_between(
        self,
        node0: int | dict[str, str | int],
        node1: int | dict[str, str | int],
        connections_lookup: ConnectionsLookup = ConnectionsLookup.REPORTED,
        node0_as: NodeIs = NodeIs.ANY,
    ) -> pl.DataFrame:
        """Report direct connections between two nodes.

        By default, look for direct connections in either direction between
        `node0` and `node1`.

        Args:
            node0:
                Index (int) or node information (dict) identifying node0.
            node1:
                Index (int) or node information (dict) identifying node1.
            connections_lookup:
                Source from which to find connections.
            node0_as:
                Role node0 should play in the connection.

        Returns:
            Matching connections (pl.DataFrame), empty if none exist.
        """
        connections_lookup = self._get_available_connection_lookup(
            connections_lookup
        )

        if connections_lookup == ConnectionsLookup.ALL:
            return self._direct_connection_from_edge_info(
                node0,
                node1,
                node0_as,
            )

        return self._direct_connection_from_network(
            node0,
            node1,
            node0_as,
        )

    def bidirectional_connections(
        self,
        node: int | dict[str, str | int],
        connections_lookup: ConnectionsLookup = ConnectionsLookup.REPORTED,
    ) -> pl.DataFrame:
        """Report all bidirectional connections of a node.

        A connection is considered bidirectional when an edge exists both from
        `node` to another node and from that node back to `node`.

        Args:
            node:
                Index (int) or node information (dict) identifying the node.
            connections_lookup:
                Source from which to find connections.

        Returns:
            Matching bidirectional connections (pl.DataFrame), empty if none
            exist.
        """

        connections_lookup = self._get_available_connection_lookup(
            connections_lookup
        )

        node_idx = (
            self._get_unique_node_index(node)
            if isinstance(node, dict)
            else node
        )

        connections_as_input, connections_as_output = self.direct_connections(
            node_idx,
            connections_lookup=connections_lookup,
        )

        bidirectional_indexes = set(connections_as_input) & set(
            connections_as_output
        )

        connection_frames = []

        for other_idx in bidirectional_indexes:
            connections = self.direct_connection_between(
                node_idx,
                other_idx,
                connections_lookup=connections_lookup,
                node0_as=NodeIs.ANY,
            )
            connection_frames.append(connections)

        if not connection_frames:
            return pl.DataFrame()

        return pl.concat(connection_frames)

    def common_connections(
        self,
        nodes: list[int],
        node_as: NodeIs = NodeIs.ANY,
        connections_lookup: ConnectionsLookup = ConnectionsLookup.REPORTED,
    ) -> list[int]:
        """
        Return internal indices of nodes directly connected to all given nodes.

         Args:
             nodes: list[int]
                 Internal indices of nodes for which to find common
                 connections.
             node_as: NodeIs
                 Whether the given nodes are inputs, outputs, or can be either.
             connections_lookup: ConnectionsLookup
                 Source from which to find connections.

         Returns:
             List with internal indices of common directly connected nodes.


        Example:
            Given:

                1 -> 3
                2 -> 3

            Nodes 1 and 2 are INPUTS to node 3.
            Node 3 is an OUTPUT of nodes 1 and 2.

            common_connections([1, 2], node_as=NodeIs.INPUT)
            -> [3]

            Given:

                3 -> 1
                3 -> 2

            Nodes 1 and 2 are OUTPUTS of node 3.
            Node 3 is an INPUT to nodes 1 and 2.

            common_connections([1, 2], node_as=NodeIs.OUTPUT)
            -> [3]

        """
        connections = []

        for node in nodes:
            inputs, outputs = self.direct_connections(
                node,
                connections_lookup=connections_lookup,
            )

            if node_as == NodeIs.INPUT:
                node_connections = inputs
            elif node_as == NodeIs.OUTPUT:
                node_connections = outputs
            else:
                node_connections = inputs + outputs

            connections.append(set(node_connections))

        common_connections = sorted(set.intersection(*connections))

        return common_connections

    def common_connections_node_info(
        self,
        nodes: list[int],
        node_as: NodeIs = NodeIs.ANY,
        connections_lookup: ConnectionsLookup = ConnectionsLookup.REPORTED,
    ) -> pl.DataFrame:
        """Return all node metadata for common directly connected nodes."""
        common_indexes = self.common_connections(
            nodes,
            node_as=node_as,
            connections_lookup=connections_lookup,
        )
        return self.node_information_from_index(common_indexes)
