# Mathematical Description of a Network

Whilst we can get by with a colloquial description of networks, in some places it will be necessary for us to use a more mathematical description of these objects.
We aim to set a standard for our notation—and conventions—in this section.

A network $G = (V, E)$ consists of:

- A set of **nodes** (also called vertices), denoted by $V$.
  The nodes are indexed by some counter, so we typically write $V := \{ v_0, v_1, ..., v_N\}$ where each $v_i$ is one node, uniquely identified by the value of $i$.
  As such, when referring to nodes it is common to drop the $v_i$ notation and simply say "node $i$" in place of "node $v_i$".
- A set of edges $E$.
  Each edge $e\in E$ is written as $e = (i, j, w)$ (or $e = (v_i, v_j, w)$).
  $(i, j, w)$ means that there is an edge (connection) directed from node $i$ into node $j$, with a weight of $w$.

The edge table that describes a network can then simply be seen as a listing of the edges $e\in E$.

:::{note}
If the (brain)-regions are provided with unique identifiers, these same unique identifiers can be used as the indexes $i$ for the mathematical description of the nodes.

However, in practice the package prefers the use of integer indexes to identify the nodes, rather than names or abbreviations that connectivity data may prefer.
Where this is important, the package keeps a record of the correspondence between "internal indexes" for the nodes and the user-facing identifiers for the brain regions they represent.
:::

<!-- FIXME: add API link for Connections.network -->

The API also separates the "metadata" concerning the nodes/regions and edges/connections from the underlying network object itself.
As such, the nodes in the `.network` use integer indexing, which is referred to as their "internal indexes".
These internal indexes are appended as a column to the node metadata, so there is a means of translating between the neuroscientific information about a brain region and its abstract representation.
Several functions also exist to aid in obtaining a selection of nodes by matching metadata criteria, and vice-versa.
