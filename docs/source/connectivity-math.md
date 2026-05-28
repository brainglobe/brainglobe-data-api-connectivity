# Mathematical Description of a Network

Whilst we can get by with a colloquial description of networks, in some places it will be necessary for us to use a more mathematical description of these objects.

We aim to set a standard for our notation and conventions in this section.

## Networks
A network—also called a *graph* in mathematical contexts—is written as $G = (V, E)$.

It consists of two components:
1. A set of **nodes** (or *vertices*), denoted by $V$.
2. A set of **edges**, denoted by $E$.

When modelling brain connectivity, the nodes ($V$) represent the **brain regions**, and the edges ($E$) represent the **connections** between them.

### 1. Nodes
The nodes are indexed by some counter, so we typically write $V := \{ v_0, v_1, ..., v_N\}$ where each $v_i$ is one node, uniquely identified by the value of $i$.

As such, when referring to nodes it is common to drop the $v_i$ notation and simply say "node $i$" in place of "node $v_i$".

### 2. Edges
Each edge $e\in E$ is written as $e = (i, j, w)$ (or $e = (v_i, v_j, w)$).

$(i, j, w)$ represents a directed, weighted edge that:

- starts **from** node $i$
- goes **to** node $j$
- has **weight** $w$

### Edge Table
The edge table that describes a network can then simply be seen as a listing of the edges $e\in E$.

For an example of how an edge table corresponds to a network, see [Example: Edge Table and Corresponding Network](./connectivity-data.md#example-edge-table-and-corresponding-network).

### Internal Indexes
If the brain regions are provided with unique identifiers, these same unique identifiers can be used as the indexes $i$ for the mathematical description of the nodes.

However, the tools used in this package prefer the use of integer indexes to identify the nodes, rather than names or abbreviations.

The API separates the metadata from the underlying network object itself. The nodes in the `.network` use integer indexing, which is referred to as their "internal indexes".

<!-- FIXME: add API link for Connections.network -->

Where this is important, the package keeps a record of the correspondence between "internal indexes" for the nodes and the user-facing identifiers for the brain regions they represent.

These "internal indexes" are appended as a column to the node metadata, so there is a means of translating between the neuroscientific information about a brain region and its abstract representation.

Several functions also exist to aid in obtaining a selection of nodes by matching metadata criteria, and vice-versa.

#### Example: Internal Indexes

When the [example edge table](./connectivity-data.md#example-edge-table-and-corresponding-network) is loaded, the package assigns each unique region label an internal integer index.

| region | internal index |
|--------|----------------|
| l‑a    | 0              |
| l‑b    | 1              |
| r‑a    | 2              |
| r‑b    | 3              |

The corresponding edge table (using internal indexes instead of region labels) becomes:

| source | target | weight |
|--------|--------|--------|
| 0      | 1      | 1.0    |
| 0      | 2      | 0.5    |
| 1      | 2      | 0.5    |
| 1      | 3      | 0.5    |
| 2      | 3      | 1.0    |
| 3      | 0      | 0.1    |
| 3      | 2      | 0.1    |
