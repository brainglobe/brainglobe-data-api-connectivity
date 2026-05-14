# Welcome to brainglobe-data-api-connectivity's documentation

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   neuro_to_math
   _gallery_examples/index
```

This package acts as an interface between neurological region-connectivity data and network (graph) analysis.
Connectivity data can be efficiently represented as a [network](./neuro_to_math.md), which means that common questions one want to run as part of an analysis of such data can be handled efficiently and quickly if one can formulate the corresponding mathematical questions and structures.
The purpose of the package is to allow users to ask these neurologically-motivated questions and receive the corresponding answers in an intuitive and helpful format; whilst in the background handling the mathematical representation so the user does not need to concern themselves with translating back and forth.

For example, the question of "what is the shortest path between brain areas A and B that passes through region C?" can be phrased as the purely mathematical task of "determine the shortest path between region A and region C, and then the shortest path between region C and region B, and concatenate them", which can be handled using well-established network algorithms.

## Connectivity Data Input Format

## Interacting with the Data
