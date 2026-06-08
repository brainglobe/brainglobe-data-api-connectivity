from enum import IntEnum, StrEnum


class NodeInConnection(StrEnum):
    """
    Options for node roles in a connection.

    At a glance:
    - EITHER : Report all connections involving a node.
    - INPUT : Report only connections that have a node as the input / source.
    - OUTPUT : Report only connections that have a node as the output / target.

    Edges (connections) are directed, which means that when one asks questions
    about the direct connections that a node has, one may optionally want to
    specify whether the node in question is the `input` (or "source") node in
    the connection, the `output` (or "target") node in the connection, or
    either.

    These options are standardised as `StrEnum`s to avoid potentially diverging
    string and language conventions in the API.
    """

    EITHER = "either"
    INPUT = "source"
    OUTPUT = "target"


class ConnectionsLookup(IntEnum):
    """
    Options flag for querying reported connections.

    At a glance:
    - ALL : Use the `.edge_info`, if it exists, to lookup connections.
    - REPORTED : Use the `.network` to lookup connections.

    The network objects that are constructed by the API use so-called
    "reported" connection data, which is essentially the information per
    connection that is deemed the most accurate or reliable. By contrast, the
    edge information attached to a `Connections` object may contain multiple
    reports for the same connection, with different levels of strength and
    accuracy of the reported result. One must decide from which source to draw
    information about the connections when posing queries about them.

    These options are standardised as `IntEnum`s to avoid potentially diverging
    conventions in the API.
    """

    ALL = 0
    REPORTED = 1
