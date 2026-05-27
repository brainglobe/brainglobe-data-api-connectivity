# Neuroscience and Mathematical Terminology

On this page we aim to provide a short summary that reconciles the terminology used to describe connectivity data in neuroscience, and map it to the corresponding mathematical terminology surrounding networks.

## Connectivity Data and Networks

Connectivity data generally consists of the following components:

- Information about the brain-regions, whose connections have been explored.
  Typically they will be given some form of unique identifier, as well as names, abbreviations, any groupings of regions, and other data like their spatial coordinates.
- A listing of the connections between regions (which are considered directed), and the "strength" of each connection given as a numerical value.
- Further information about the connections.
  This may consist of information about the studies in which they were examined, any special reference names they might have, or any user- or neurological-groupings of the connections themselves.
- Any metadata about the dataset, or study that generated the dataset.
  These may be fields such as the species, sex, or age of the animal from which the connectivity information was obtained, the institute at which the experiment was conducted, etc.

Mathematical networks are an abstract representation of connections between a collection of objects.
A network consists of a collection of **nodes** that are connected to each other by (directed) **edges**, with each edge possessing a **weight** that characterises the connection.
As such, if we have some connectivity data;

- The brain regions are represented by the nodes.
  If the brain regions are provided with unique identifiers, these same identifiers can be used for the nodes.
- The connections between the brain regions are represented by edges.
- The strength of the connections are represented by the edge weights.

The network that represents a particular dataset can be fully described simply by listing the connections, in what is often called an **edge list** or **edge table**.
This is what our API expects to receive as an input.
An edge table is just a three-column CSV file, where each row specifies one connection in the form

```text
source node, target node, connection strength
```

:::{note}
Any further information about the regions can be stored in some suitable data format (such as a `DataFrame`) that allows indexing via the region identifiers - this information is referred to as "region metadata" or "node metadata".

Similarly, any further information about the connections (excluding their strengths) can be stored in tabular format that allows indexing via **pairs of** region identifiers that describes the connection's start and end - this information is referred to as "connection metadata" or "edge metadata".

Region metadata (save for provision a unique identifiers for the regions) and connection metadata is entirely optional from a purely mathematical analysis of the connectivity data.
However, such information is normally incredibly helpful to have on-hand when one wants to run or plan neuroscientifically-relevant analysis!
:::

### Example: Connectivity Data and the Corresponding Network

FIXME: Get Stella to actually suggest sensible names for the brain regions & column names.

Let us suppose we have gathered the following connectivity dataset, that describes connections between two brain regions on either side of a brain:

| name | abbr | side of brain |
| :-: | :-: | :-: |
| alpha | l-a | left |
| bravo | l-b | left |
| alpha | r-a | right |
| bravo | r-b | right |

where the `abbr` column is a unique identifier that we have given to each region in the brain.
Suppose we have then identified the following information about the connections between these regions:

| source region | target region | connection strength | observation notes |
| :-: | :-: | :-: | :-: |
| l-a | l-b | 1.0 | high confidence |
| l-a | r-a | 0.5 | high confidence |
| l-b | r-a | 0.5 | low confidence |
| l-b | r-b | 0.5 | high confidence |
| r-a | r-b | 1.0 | med confidence |
| r-b | l-a | 0.1 | high confidence |
| r-b | l-b | 0.0 | not present |
| r-b | r-a | 0.1 | low confidence |

Our connectivity information can be used to construct our edge-table CSV file:

```text
l-a, l-b, 1.0
l-a, r-a, 0.5
l-b, r-a, 0.5
l-b, r-b, 0.5
r-a, r-b, 1.0
r-b, l-a, 0.1
r-b, r-a, 0.1
```

This edge table fully describes the network that will represent our connectivity data; there will be four nodes (`l-a`, `l-b`, `r-a`, `r-b`) and a total of seven edges.
We can even illustrate this network:

![An illustration of the example connectivity data, as a network.](./_static/example-connectivity-network.svg)

and we can also make a few observations:

- The connection between `l-a` and `l-b` is bidirectional.
  However, the connection from `l-a` to `l-b` is stronger (strength 1) than the connection from `l-b` to `l-a` (strength 0.5).
- The connection between `r-b` and `l-a` is unidirectional, directed from `r-b` into `l-a`.
  It is also one of the weakest connections in the network, with a strength of 0.1.
- Each edge in the edge list corresponds to a **direct** connection between the regions.
- In addition to the direct connections, all of the regions are indirectly connected to each other (since there is a path that can be taken to each any particular region from any other region).

:::{note}
Connections that are excluded from the edge table can fall into several categories:

- No data: Following an extensive search, no reports were found with data relevant to the connection and suitable for inclusion in the network analysis.
- Unclear: It is unclear from the report if the connection exists or not. It is reasonable to infer that the connection is likely weak (at most), very weak, or absent.
- Evidence of Absence: Evidence indicates that the connection does not exist.
- Same origin & termination: Within‑region connections are not included in the network.

In our example, we have excluded the `r-b` to `l-b` connection on the grounds of our experiments determining that no such connection exists between these two brain regions.

Note that a connection being excluded from the edge table (and thus the network) has **different implications** to including a connection between two regions with a strength of 0.
:::

## Mathematical Details and Implementation

Whilst we can get by with a colloquial description of networks, in some places it will be necessary for us to use a more mathematical description of these objects.
We aim to set a standard for our notation - and conventions - in this section.

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

In practice, the API relies on the [`rustworkx`](https://www.rustworkx.org/) library for all network-related queries.
Connectivity data is represented by a [`PyDiGraph` class](https://www.rustworkx.org/apiref/rustworkx.PyDiGraph.html), which is exposed through the `Connections.network` attribute to allow for flexible querying of the network structure if specialised analyses are required.

<!-- FIXME: add API link for Connections.network -->

The API also separates the "metadata" concerning the nodes/regions and edges/connections from the underlying network object itself.
As such, the nodes in the `.network` use integer indexing, which is referred to as their "internal indexes".
These internal indexes are appended as a column to the node metadata, so there is a means of translating between the neuroscientific information about a brain region and its abstract representation.
Several functions also exist to aid in obtaining a selection of nodes by matching metadata criteria, and vice-versa.
