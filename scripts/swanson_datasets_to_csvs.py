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

morph_dict = {"one": 1, "two": 2}


def _morph(region_side: str) -> int:
    return morph_dict[region_side]


if __name__ == "__main__":
    data_folder = Path("data")

    # File paths
    matrix_file = data_folder / "swansonDatasetS3 CNS data matrices JHr1.xlsx"
    metadata_file = data_folder / "swansonDatasetS2 CNS CRs JHr1.xlsx"

    # Sheet and range definitions
    matrix_sheets = [f"CNS2{sex} modules" for sex in ["m", "f"]]
    data_range = ("T8", "AFU841")
    info_range = ("A7", "S841")

    #  MRCC (multiresolution consensus clustering number)
    mrcc_col = "P"  # MRCC
    mrcc_row = 5  # MRCC

    # Save metadata CSV
    pd.read_excel(metadata_file).to_csv(
        data_folder / "edge_metadata.csv", index=False
    )

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

        # Morph region identifiers
        region_info = {"region_side": "one", "region_abbr": "STN"}
        region_to_node_heading = {"region_side": "side", "region_abbr": "abbr"}
        morph_value = {"region_side": _morph}
        convert.lookup_and_morph_node_index(
            region_info, info, region_to_node_heading, morph_value
        )

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
        pattern="*_info.csv",
        folder=data_folder,
        output_name="node_info.csv",
    )
