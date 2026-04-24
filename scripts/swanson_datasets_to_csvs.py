import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from brainglobe_data_api_connectivity.preprocess.validate_data import (
    validate_matrix,
)


def get_cell_range(
    cell_range: tuple[str, str],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Convert two Excel A1 references into (col_range, row_range),
    where each range is a (start, end) pair of zero‑based indices.
    """
    if len(cell_range) != 2:
        raise ValueError(
            "cell_range must contain exactly two cell references."
        )

    start_col, start_row = split_cell_reference(cell_range[0])
    end_col, end_row = split_cell_reference(cell_range[1])

    return (start_col, end_col), (start_row, end_row)


def split_cell_reference(ref):
    """
    Split an Excel A1-style cell reference into (column_label, row_number).
    Example: 'AA841' → (26, 841)
    """
    col_ref = "".join(filter(str.isalpha, ref))
    row_ref = "".join(filter(str.isdigit, ref))

    if not col_ref or not row_ref:
        raise ValueError(f"Invalid cell reference: {ref}")

    return column_reference_to_index(col_ref), int(row_ref)


def column_reference_to_index(label):
    """
    Convert Excel column labels to a 0-based index.
    A=0, B=1, ..., Z=25, AA=26, AB=27, ..., AZ=51, BA=52, ..., ZZZ=18277
    """
    label = label.upper()

    if not label.isalpha():
        raise ValueError(f"Invalid column label: {label}")

    index = 0
    for char in label:
        index = index * 26 + (ord(char) - ord("A") + 1)

    return index - 1


def get_df_from_excel(file, sheet_name, data_range, header=None):
    """Return DataFrame sliced to given row/column ranges."""
    col_range, row_range = get_cell_range(data_range)

    start_row, end_row = row_range
    start_col, end_col = col_range

    skiprows = start_row - 1
    nrows = end_row - start_row + 1
    usecols = list(range(start_col, end_col + 1))

    col_range, row_range = get_cell_range(data_range)

    df = pd.read_excel(
        file,
        sheet_name=sheet_name,
        skiprows=skiprows,
        nrows=nrows,
        usecols=usecols,
        header=header,
    )
    return df


def check_ids(file_path, sheet_name, data_range, row_with_ids, col_with_ids):
    """Specific to this dataset, checks that row and col ids match and that
    they are a range from 1 to n."""
    row_ids = get_row_ids(file_path, sheet_name, data_range, row_with_ids)
    col_ids = get_col_ids(file_path, sheet_name, data_range, col_with_ids)
    if row_ids != col_ids:
        raise ValueError("Row and column IDs do not match.")

    expected_ids = list(range(1, len(row_ids) + 1))
    if row_ids != expected_ids:
        raise ValueError(
            f"Row IDs {row_ids} do not match expected {expected_ids}."
        )
    return row_ids, col_ids


def get_row_ids(file, sheet, data_range, col_label):
    (_, _), (r0, r1) = get_cell_range(data_range)
    col = column_reference_to_index(col_label)
    return (
        pd.read_excel(
            file,
            sheet_name=sheet,
            skiprows=r0 - 1,
            nrows=r1 - r0 + 1,
            usecols=[col],
            header=None,
        )
        .iloc[:, 0]
        .tolist()
    )


def get_col_ids(file, sheet, data_range, row_num):
    (c0, c1), (_, _) = get_cell_range(data_range)
    return (
        pd.read_excel(
            file,
            sheet_name=sheet,
            skiprows=row_num - 1,
            nrows=1,
            usecols=list(range(c0, c1 + 1)),
            header=None,
        )
        .iloc[0]
        .tolist()
    )


def rename_columns(columns):
    return (
        columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("\n", "_")
        .str.replace(".", "")
        .str.replace("#", "")
    )


def consolidate_info_files(data_folder):
    """Load all info files and check whether they are the same.
    Remove duplicates and rename the first one to simply info.csv."""

    info_path = os.path.join(data_folder, "node_info.csv")
    if os.path.exists(info_path):
        os.remove(info_path)

    info_files = [f for f in os.listdir(data_folder) if f.endswith("info.csv")]
    for i in range(len(info_files)):
        for j in range(i + 1, len(info_files)):
            info_i = pd.read_csv(os.path.join(data_folder, info_files[i]))
            info_j = pd.read_csv(os.path.join(data_folder, info_files[j]))
            if not info_i.equals(info_j):
                raise ValueError(
                    "Info files {0} and {1} do not match.".format(
                        info_files[i], info_files[j]
                    )
                )

    os.rename(
        os.path.join(data_folder, info_files[0]),
        os.path.join(data_folder, "node_info.csv"),
    )

    for f in info_files:
        if f != info_files[0]:
            os.remove(os.path.join(data_folder, f))


def convert_matrix_to_edge_table(m: pd.DataFrame):
    adjacency_matrix = m.to_numpy()
    connections = adjacency_matrix.nonzero()
    weights = adjacency_matrix[connections]
    edge_table = np.column_stack(connections + (weights,))
    return edge_table


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


morph_dict = {"one": 1, "two": 2}


def _morph(region_side):
    return morph_dict[region_side]


if __name__ == "__main__":
    data_folder = "data"
    matrices = "swansonDatasetS3 CNS data matrices JHr1.xlsx"
    matrix_sheets = [f"CNS2{sex} modules" for sex in ["m", "f"]]
    data_range = ("T8", "AFU841")
    info_range = ("A7", "S841")

    file_path = os.path.join(data_folder, matrices)

    metadata = "swansonDatasetS2 CNS CRs JHr1.xlsx"
    metadata_file = os.path.join(data_folder, metadata)

    # df = pd.read_excel(metadata_file)
    # df.to_csv(os.path.join(data_folder, "edge_metadata.csv"), index=False)

    for sheet in matrix_sheets:
        if sheet not in pd.ExcelFile(file_path).sheet_names:
            raise ValueError(f"Sheet {sheet} not found in {file_path}.")
        matrix = get_df_from_excel(
            file_path, sheet_name=sheet, data_range=data_range
        )
        row_ids, col_ids = check_ids(file_path, sheet, data_range, "P", 5)
        validate_matrix(matrix, row_ids, col_ids)

        # convert m to edge table
        edge_table = convert_matrix_to_edge_table(matrix)

        info = get_df_from_excel(
            file_path, sheet_name=sheet, data_range=info_range, header=0
        )
        info.rename(columns={info.columns[0]: "Side"}, inplace=True)
        info.columns = rename_columns(info.columns)

        region_info = {"region_side": "one", "region_abbr": "STN"}
        region_to_node_heading = {"region_side": "side", "region_abbr": "abbr"}
        morph_value = {"region_side": _morph}

        idx = lookup_node_index(
            region_info, info, region_to_node_heading, morph_value
        )

        np.savetxt(
            os.path.join(
                data_folder, sheet.replace(" ", "_") + "_edge_table.csv"
            ),
            edge_table,
            fmt="%d",
            delimiter=",",
        )

        info.to_csv(
            os.path.join(data_folder, sheet.replace(" ", "_") + "_info.csv"),
            index=False,
        )

    consolidate_info_files(data_folder)

    pass
