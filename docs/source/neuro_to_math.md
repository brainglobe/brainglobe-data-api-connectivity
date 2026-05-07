# Neuroscience and Mathematical Terminology

On this page we aim to provide a short summary that reconciles the terminology used to describe connectivity data in neuroscience, and map it to the corresponding mathematical terminology surrounding networks.

## Networks

A network (or graph) consists of:

- A set of nodes (also called vertices) $V$.
  The nodes are indexed by some counter, so we typically write $V := \{ v_0, v_1, ..., v_N\}$ where each $v_i$ is one node, uniquely identified by the value of $i$.
  As such, when referring to nodes it is common to drop the $v_i$ notation and simply say "node $i$" in place of "node $v_i$".
- A set of edges $E$.
  Each edge $e\in E$ is written as $e = (i, j, w)$ (or $e = (v_i, v_j, w)$ if you prefer to keep the explicit node notation).
  The edge $(i, j, w)$ means that there is an edge (connection) direct from node $i$ into node $j$, with a weight of $w$.

So long as one has a consistent indexing for the nodes, a network can then be built up by listing the edges in $E$.
Such a listing is often referred to as an "edge list" or "edge table", which is simply a 3-column format where each row lists a single edge $e$ in $(i, j, w)$ format.

## Connectivity Data

Connectivity data generally consists of three components:

- Information about the brain-regions, whose connections have been explored.
  Typically they will be given some form of unique identifier, as well as names, abbreviations, any groupings of regions, and other data like their spatial coordinates.
- A listing of the connections between regions, which are considered directed.
  A connection is further attributed some quantifiable "weight"; which may represent the strength of that connection, the time for a signal to travel along the connection, or other neurological quantity of interest.
- Further information about the connections.
  This may consist of information about the studies in which they were examined, any special reference names they might have, or any user- or neurological-groupings of the connections themselves.

## Reconciling the Terms

As such, the regions should be thought of as nodes, and the connections as edges.
If the regions are provided with unique identifiers, these same unique identifiers can be used as the indexes $i$ for the mathematical description of the nodes.
In the same way, this also means that list of connections between the regions acts as out edge table, and thus we can build a network to represent our connectivity data.
Any further information about the regions can be stored in some suitable data format (such as a `DataFrame`) that allows indexing via the region identifiers - this information is referred to as "node/region metadata".
Similarly, any further information about the connections (excluding their weights) can be stored in tabular format that allows indexing via the $(i, j)$ pair that describes the connection's start and end - this information is referred to as "edge metadata".

:::{note}
By convention, regions that are not connected do not appear in the edge list.
This is to distinguish regions that aren't connected, from regions that may happen to be connected with a weight of $0$, for example.

However, edge metadata _may_ contain information about a pair of regions $(i, j)$ that are not connected, without issue.
This is common when, for example, a study has determined that two regions are not connected and needs to encode this information into the connectivity dataset.
As such, not every connection that appears in the edge metadata may have an edge representing it in the mathematical network, however every edge in the mathematical network _does_ have a corresponding entry in the edge metadata table.
:::

:::{note}
Node metadata (save for provision of a unique identifier for the regions) and edge metadata are entirely optional from a purely mathematical analysis of the connectivity data.
However, such information is normally incredibly helpful to have on-hand when one wants to run or plan neurologically-relevant analysis!
:::
