# Connectivity Data
We can divide connectivity data into:
- a network consisting of nodes, edges, and weights
- the metadata (everything we know about the regions, connections, and dataset)

While graph analysis can be done using only the network, the metadata is essential for guiding analyses, understanding results, and making the work reproducible.

## Networks

Mathematical networks are an abstract representation of connections between a collection of objects.
A network consists of a collection of **nodes** that are connected to each other by (directed) **edges**, with each edge possessing a **weight** that characterises the connection.
As such, if we have some connectivity data;

- The brain regions are represented by the nodes.
  If the brain regions are provided with unique identifiers, these same identifiers can be used for the nodes.
- The connections between the brain regions are represented by edges.
- The strength of the connections are represented by the edge weights.

The network that represents a particular dataset can be fully described simply by listing the connections, in what is often called an **edge list** or **edge table**.

This is what our API expects to receive as an input.

An edge table is just a three-column CSV file, where each row specifies one connection in the form:

```text
source node, target node, connection strength
```

### Example: Edge Table and Corresponding Network

Our connectivity information can be used to construct our edge table CSV file:

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

We can illustrate this network:

![An illustration of the example connectivity data, as a network.](./_static/example-connectivity-network.svg)

and we can also make a few observations:

- The connection between `l-a` and `l-b` is bidirectional.
  However, the connection from `l-a` to `l-b` is stronger (strength 1) than the connection from `l-b` to `l-a` (strength 0.5).
- The connection between `r-b` and `l-a` is unidirectional, directed from `r-b` into `l-a`.
  It is also one of the weakest connections in the network, with a strength of 0.1.
- Each edge in the edge list corresponds to a **direct** connection between the regions.
- In addition to the direct connections, all of the regions are indirectly connected to each other (since there is a path that can be taken to each any particular region from any other region).

:::{note}
Often the description of nodes, edges, and weights given here is all you need. But sometimes you may need a more precise, mathematical description of how nodes, edges, and weights are defined, for this see [Mathematical Description of a Network](..\connectivity-math.md).
:::

## Metadata
Although a great deal can be learned from the network structure alone, to make sense of what the nodes and edges actually represent you also need the metadata.

Connectivity metadata generally consists of the following components:
- **Dataset Metadata**: Information about the dataset, or study that generated the dataset.
  These may be fields such as the species, sex, or age of the animal from which the connectivity information was obtained, the institute at which the experiment was conducted, etc.
- **Node Metadata**: Information about the brain-regions, whose connections have been explored.
  Typically they will be given some form of unique identifier, as well as names, abbreviations, any groupings of regions, and other data like their spatial coordinates.
- **Edge Metadata**: Information about the connections.
  This may consist of information about the studies in which they were examined, any special reference names they might have, or any user- or neurological-groupings of the connections themselves.

### Example: Metadata

**Dataset Metadata**

|         |                  |
| ------------ | --------------------- |
| project      | Alpha–Bravo Connectivity |
| contributors | Jane Doe              |
| year         | 2024                  |
| species      | Rattus bravalpha   |
| age          | 12 weeks              |
| sex          | F                     |


**Node Metadata**

| name  | abbr | side of brain |
| ----- | -----| -------------- |
| alpha | l‑a  | left           |
| bravo | l‑b  | left           |
| alpha | r‑a  | right          |
| bravo | r‑b  | right          |

**Note**: The `abbr` column provides the unique identifiers used in the edge table.

**Edge Metadata**

| source | target | method     | reference        | notes |
| ------ | ------ | ---------- | ---------------- | ----- |
| l‑a    | l‑b    | anterograde | Doe & Roe, 1992  |       |
| l‑a    | r‑a    | retrograde | Doe & Roe, 1992  |       |
| l‑b    | r‑a    | anterograde | Doe et al., 1999 |       |
| l‑b    | r‑b    | retrograde | Doe et al., 1999 |       |
| r‑a    | r‑b    | anterograde | Doe et al., 1999 |       |
| r‑b    | l‑a    | retrograde | Li et al., 2012  |       |
| r‑b    | l‑b    | anterograde | Doe et al., 1996 | injection too large to interpret positive labeling; evidence used for “absent” classification |
| r‑b    | r‑a    | retrograde | Li et al., 2012  |       |

**Note**: The `source` and `target` columns provide the unique identifiers used in the edge table.

## Identifiers

It is important to be able to match the metadata to the network structure so that each node and connection in the edge table can be related back to the biological regions and studies they represent.

In this example, the common identifiers used the abbreviations `l‑a`, `l‑b`, `r‑a`, and `r‑b`. These appear in the edge table as the node labels, and the same labels appear in the node metadata so that each entry in the network can be matched to the corresponding brain region.
