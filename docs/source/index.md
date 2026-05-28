# Welcome to brainglobe-data-api-connectivity's documentation

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   connectivity-math
   connectivity-data
   examples/index
```

This package acts as an interface between brain connectivity data and graph‑based analyses.

The purpose of the package is to provide neuroscientists with an interface for asking questions about a connectivity network without having to translate those questions into mathematical graph operations.

For example, the question "what is the shortest path between brain areas A and B that passes through region C?" translates to "determine the shortest path between region A and region C, and then the shortest path between region C and region B, and concatenate them", which can be handled using well-established network algorithms.

In practice, the API relies on the [`rustworkx`](https://www.rustworkx.org/) library for all network-related queries. Connectivity data is represented by a [`PyDiGraph` class](https://www.rustworkx.org/apiref/rustworkx.PyDiGraph.html), which is exposed through the `Connections.network` attribute to allow for flexible querying of the network structure if specialised analyses are required.

## Connectivity Data
See [Connectivity Data](./connectivity-data.md) for details on the structure of the input data, including networks and metadata.

## Interacting with the Data
