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
    "matrix_sheets": [f"CNS2{sex} modules (raw)" for sex in ["m", "f"]],
    "matrix_range": ("T8", "AFU841"),
    "node_info_range": ("A7", "S841"),
    "mrcc_col": "P",
    "mrcc_row": 5,
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

    edge_info.to_csv(DATA_FOLDER / "edge_info.csv", index=False)

    # Process each matrix sheet
    for sheet in SWANSON_PARAMS["matrix_sheets"]:
        # Load and validate matrix and ids
        matrix = excel.get_df_from_excel(
            SWANSON_PARAMS["matrix_file"],
            sheet,
            SWANSON_PARAMS["matrix_range"],
        )
        validate_input.validate_adjacency_matrix(matrix)

        # Convert matrix to edge table
        edge_table = convert.convert_matrix_to_edge_table(matrix)

        # Load and tidy node info
        node_info = excel.get_df_from_excel(
            SWANSON_PARAMS["matrix_file"],
            sheet_name=sheet,
            data_range=SWANSON_PARAMS["node_info_range"],
            header=0,
        )
        node_info.rename(columns={node_info.columns[0]: "Side"}, inplace=True)
        node_info.columns = tidy.rename_columns(node_info.columns)

        # Save outputs
        sheet_tag = sheet.replace(" ", "_")
        np.savetxt(
            DATA_FOLDER / f"{sheet_tag}_edge_table.csv",
            edge_table,
            fmt="%d",
            delimiter=",",
        )
        node_info.to_csv(DATA_FOLDER / f"{sheet_tag}_info.csv", index=False)

    # Consolidate duplicates
    target = DATA_FOLDER / "node_info.csv"
    target.unlink(missing_ok=True)
    files = sorted(DATA_FOLDER.glob("[!edge]*_info.csv"))
    if files:
        reference = pd.read_csv(files[0])
        if any(not pd.read_csv(f).equals(reference) for f in files[1:]):
            raise ValueError("Info files do not match.")
        files[0].rename(target)
        for f in files[1:]:
            f.unlink()
