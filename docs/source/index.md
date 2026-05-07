# Welcome to brainglobe-data-api-connectivity's documentation

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
```

This package acts as an interface between neurological region-connectivity data and network (graph) analysis.
Connectivity data generally consists of three components:

- Information about the brain-regions, whose connections have been explored.
  Typically their names, abbreviations, any groupings of regions, and other data like their spatial coordinates.
- The strength of the inter-region connections, which are considered directed.
  This means that region A may connect to region B, but there may not be a reverse connection.
  A connection is further attributed a weight, which may represent the strength of that connection, the time for a signal to travel along the connection, or other neurological quantity of interest.
- Further information about the connections.
  This may consist of information about the studies in which they were examined, any special reference names they might have, or any user- or neurological-groupings of the connections themselves.

Mathematically, such connection information is represented as a network (or graph), which makes it tractable to analysis via algorithms or manipulations of such objects.
The regions are represented by the nodes of the graph, and the connections by the edges within the graph.
This allows us to phrase questions such as "what is the minimum travel time for a signal from region A to reach region B" as purely mathematical questions ("find the path between region A and region B that minimises the signal travel time") which can be solved using well-established network algorithms.

## Connectivity Data Input Format

## Interacting with the Data
