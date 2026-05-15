"""Extracts, validates, and reformats data from Excel files into
standardised CSVs required for API input.

This script reads the connectivity matrices and associated metadata from the
Excel sources, validates that the data matches the expected structure, and
outputs standardised metadata and edge‑table CSV files. The CSVs load more
quickly, are easier to work with, and can be regenerated reliably whenever the
Excel files are updated.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from brainglobe_data_api_connectivity.io import excel, validate_input
from brainglobe_data_api_connectivity.utils import convert, tidy

EDGE_INFO_MORPH_DICT = {
    "side": {"one": int(1), "two": int(2), "left": int(1), "right": int(2)}
}

if __name__ == "__main__":
    data_folder = Path("data")

    # File paths
    matrix_file = data_folder / "swansonDatasetS3 CNS data matrices JHr1.xlsx"
    edge_information_file = data_folder / "swansonDatasetS2 CNS CRs JHr1.xlsx"

    # Sheet and range definitions
    matrix_sheets = [f"CNS2{sex} modules" for sex in ["m", "f"]]
    data_range = ("T8", "AFU841")
    info_range = ("A7", "S841")

    #  MRCC (multiresolution consensus clustering number)
    mrcc_col = "P"  # MRCC
    mrcc_row = 5  # MRCC

    # Clean and save edge_information CSV
    edge_info = pd.read_excel(edge_information_file, header=[0, 1])
    edge_info.columns = edge_info.columns.map(
        lambda x: "_".join([str(i) for i in x if "Unnamed:" not in i])
    )  # combine multi-index headers
    edge_info.columns = tidy.rename_columns(edge_info.columns)

    # different `region_side` labels are used ("one" "two" / "left" "right")
    # mapping on `node_info` sides 1 and 2
    morph_dict = EDGE_INFO_MORPH_DICT["side"]

    for col in [c for c in edge_info.columns if "side" in c]:
        edge_info[col] = edge_info[col].map(morph_dict)
    edge_info.to_csv(data_folder / "edge_info.csv", index=False)

    for sheet in matrix_sheets:
        # Load and validate matrix and ids
        matrix = excel.get_df_from_excel(matrix_file, sheet, data_range)
        mrcc_row_values = excel.get_row_values(
            matrix_file, data_range, mrcc_col, sheet
        )
        mrcc_col_values = excel.get_col_values(
            matrix_file, data_range, mrcc_row, sheet
        )
        validate_input.check_ids(mrcc_row_values, mrcc_col_values)
        validate_input.validate_matrix(
            matrix, mrcc_row_values, mrcc_col_values
        )

        # Convert matrix to edge table
        edge_table = convert.convert_matrix_to_edge_table(matrix)

        # Load and tidy node info
        info = excel.get_df_from_excel(
            matrix_file, sheet_name=sheet, data_range=info_range, header=0
        )
        info.rename(columns={info.columns[0]: "Side"}, inplace=True)
        info.columns = tidy.rename_columns(info.columns)

        # Save outputs
        sheet_tag = sheet.replace(" ", "_")
        np.savetxt(
            data_folder / f"{sheet_tag}_edge_table.csv",
            edge_table,
            fmt="%d",
            delimiter=",",
        )
        info.to_csv(data_folder / f"{sheet_tag}_info.csv", index=False)

    # Consolidate duplicate info files
    tidy.consolidate_duplicates(
        pattern="[!edge]*_info.csv",
        folder=data_folder,
        output_name="node_info.csv",
    )
