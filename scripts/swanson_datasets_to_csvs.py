"""Extracts, validates, and reformats data from Excel files into
standardised CSVs required for API input.

This script reads the connectivity matrices and associated metadata from the
Excel sources, validates that the data matches the expected structure, and
outputs standardised metadata and edge‑table CSV files. The CSVs load more
quickly, are easier to work with, and can be regenerated reliably whenever the
Excel files are updated.
"""

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from brainglobe_data_api_connectivity.io import excel, validate_input
from brainglobe_data_api_connectivity.utils import convert, tidy


class SwansonParams(TypedDict):
    matrix_file: Path
    edge_info_file: Path
    matrix_sheets: list[str]
    matrix_range: tuple[str, str]
    node_info_range: tuple[str, str]
    mrcc_col: str
    mrcc_row: int


EDGE_INFO_MORPH_DICT = {"side": {"one": 1, "two": 2, "left": 1, "right": 2}}
DATA_FOLDER = Path("data")
SWANSON_PARAMS: SwansonParams = {
    "matrix_file": DATA_FOLDER
    / "swansonDatasetS3 CNS data matrices JHr1.xlsx",
    "edge_info_file": DATA_FOLDER / "swansonDatasetS2 CNS CRs JHr1.xlsx",
    "matrix_sheets": [f"CNS2{sex} modules" for sex in ["m", "f"]],
    "matrix_range": ("T8", "AFU841"),
    "node_info_range": ("A7", "S841"),
    "mrcc_col": "P",
    "mrcc_row": 5,
}

RAW_CONNECTION_STRENGTH = {
    "identity": 0,
    "no article": 1,
    "unclear": 2,
    "absent": 3,
    "ax.pass.": 4,
    "v.weak": 5,
    "weak": 6,
    "weak-mod": 7,
    "exists": 8,
    "moderate": 9,
    "mod-strong": 10,
    "strong": 11,
    "v.strong": 12,
}

PROCESSED_CONNECTION_STRENGTH = {
    "identity": 0,
    "no article": 0,
    "unclear": 0,
    "absent": 0,
    "ax.pass.": 2,
    "v.weak": 1,
    "weak": 2,
    "weak-mod": 3,
    "exists": 4,
    "moderate": 4,
    "mod-strong": 5,
    "strong": 6,
    "v.strong": 7,
}

if __name__ == "__main__":
    # Clean and save edge_information CSV
    edge_info = pd.read_excel(SWANSON_PARAMS["edge_info_file"], header=[0, 1])
    edge_info.columns = edge_info.columns.map(
        lambda x: "_".join([str(i) for i in x if "Unnamed:" not in i])
    )
    edge_info.columns = tidy.rename_columns(edge_info.columns)

    morph_dict = EDGE_INFO_MORPH_DICT["side"]
    for col in [c for c in edge_info.columns if "side" in c]:
        edge_info[col] = edge_info[col].map(morph_dict)

    # Add unique region_ids
    edge_info["source_region_id"] = (
        edge_info["connection_origin_region_abbr"]
        + "_"
        + edge_info["connection_origin_region_side"].astype(str)
    )

    edge_info["target_region_id"] = (
        edge_info["connection_termination_region_abbr"]
        + "_"
        + edge_info["connection_termination_region_side"].astype(str)
    )

    edge_info["edge_id"] = (
        edge_info["source_region_id"] + "-" + edge_info["target_region_id"]
    )

    # Map raw_connection_strength
    edge_info["raw_connection_strength"] = (
        edge_info["connection_reported_value"]
        .str.lower()
        .map(RAW_CONNECTION_STRENGTH)
    )

    # Map raw_connection_strength
    edge_info["processed_connection_strength"] = (
        edge_info["connection_reported_value"]
        .str.lower()
        .map(PROCESSED_CONNECTION_STRENGTH)
    )

    edge_info.to_csv(DATA_FOLDER / "edge_info.csv", index=False)

    # Process each matrix sheet
    for matrix_id in ["CNS2f", "CNS2m"]:
        sheet = [
            sheet
            for sheet in SWANSON_PARAMS["matrix_sheets"]
            if matrix_id in sheet
        ][0]
        # Load and validate matrix and ids
        raw_matrix = excel.get_df_from_excel(
            SWANSON_PARAMS["matrix_file"],
            f"{sheet} (raw)",
            SWANSON_PARAMS["matrix_range"],
        )

        # Load and validate matrix and ids
        processed_matrix = excel.get_df_from_excel(
            SWANSON_PARAMS["matrix_file"],
            f"{sheet}",
            SWANSON_PARAMS["matrix_range"],
        )

        validate_input.validate_adjacency_matrix(raw_matrix)
        validate_input.validate_adjacency_matrix(processed_matrix)

        # Convert matrix to edge table
        edge_table_raw = convert.convert_matrix_to_edge_table(
            raw_matrix, include_zeros=True
        )

        edge_table_processed = convert.convert_matrix_to_edge_table(
            processed_matrix
        )

        # Load and tidy node info
        node_info = excel.get_df_from_excel(
            SWANSON_PARAMS["matrix_file"],
            sheet_name=sheet,
            data_range=SWANSON_PARAMS["node_info_range"],
            header=0,
        )
        node_info.rename(columns={node_info.columns[0]: "Side"}, inplace=True)
        node_info.columns = tidy.rename_columns(node_info.columns)

        # add unique region_id for nodes
        node_info["region_id"] = (
            node_info["abbr"] + "_" + node_info["side"].astype(str)
        )

        output_folder = DATA_FOLDER / matrix_id
        output_folder.mkdir(exist_ok=True)

        # Save outputs
        np.savetxt(
            output_folder / f"{matrix_id}_edge_table.csv",
            edge_table_processed,
            fmt="%d",
            delimiter=",",
        )

        node_info.to_csv(output_folder / f"{matrix_id}_info.csv", index=False)

        if matrix_id == "CNS2m":
            edge_info_sheet = edge_info[edge_info["male_or_female"] == "male"]
        elif matrix_id == "CNS2f":
            edge_info_sheet = edge_info[
                edge_info["male_or_female"] == "female"
            ]

        edge_info_sheet.to_csv(
            output_folder / f"{matrix_id}_edge_info.csv", index=False
        )
