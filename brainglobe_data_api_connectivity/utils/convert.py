from typing import Any, Callable

import numpy as np
import pandas as pd


def lookup_node_index(
    region_info: dict[str, Any],
    node_info: pd.DataFrame,
    region_to_node_heading: dict[str, str],
    morph_value: dict[str, Callable],
):

    possible_matches = node_info

    for region_heading, value in region_info.items():
        if region_heading in morph_value:
            conversion_function = morph_value[region_heading]
            value = conversion_function(value)
        if region_heading in region_to_node_heading:
            node_heading = region_to_node_heading[region_heading]
        else:
            node_heading = region_heading

        possible_matches = possible_matches[
            possible_matches[node_heading] == value
        ]

    if possible_matches.shape[0] == 1:
        return possible_matches.index[0]
    else:
        raise ValueError(
            f"Found {possible_matches.shape[0]} matches, expected 1."
        )


def convert_matrix_to_edge_table(m: pd.DataFrame) -> np.ndarray:
    adjacency_matrix = m.to_numpy()
    connections = adjacency_matrix.nonzero()
    weights = adjacency_matrix[connections]
    edge_table = np.column_stack(connections + (weights,))
    return edge_table
