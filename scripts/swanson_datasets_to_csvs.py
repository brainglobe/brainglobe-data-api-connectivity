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
        edge_table_raw = convert.convert_matrix_to_edge_table(matrix)

        # Normalise: convert 0/1/2/3 to 0
        edge_table_norm = edge_table_raw.copy()
        edge_table_norm[np.isin(edge_table_norm[:, 2], range(4)), 2] = 0

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

        # Save outputs
        sheet_tag = sheet.replace(" ", "_")
        np.savetxt(
            DATA_FOLDER / f"{sheet_tag}_edge_table.csv",
            edge_table_norm,
            fmt="%d",
            delimiter=",",
        )
        node_info.to_csv(DATA_FOLDER / f"{sheet_tag}_info.csv", index=False)

    # Add edge_id to edge_table_raw
    edge_id = (
        node_info["region_id"].iloc[edge_table_raw[:, 0]].values
        + "-"
        + node_info["region_id"].iloc[edge_table_raw[:, 1]].values
    )

    edge_table_raw_df = pd.DataFrame(
        {
            "source_region_idx": edge_table_raw[:, 0],
            "target_region_idx": edge_table_raw[:, 1],
            "raw_connection_strength": edge_table_raw[:, 2],
            "edge_id": edge_id,
        }
    )

    merged = edge_info.merge(edge_table_raw_df, on="edge_id", how="left")

    #########################################################################
    #######################       TEST           ############################
    #########################################################################

    # TODO investigate whether ids are as expected and why merged
    #  is not as expected

    edge_info_ids = set(edge_info["edge_id"])
    raw_ids = set(edge_table_raw_df["edge_id"])

    matches = edge_info_ids & raw_ids
    missing_in_raw = edge_info_ids - raw_ids
    missing_in_info = raw_ids - edge_info_ids

    print("Matches:", len(matches))
    print("Missing in raw:", len(missing_in_raw))
    print("Missing in edge_info:", len(missing_in_info))

    node_ids = set(node_info["region_id"])
    info_ids = set(edge_info["source_region_id"]) | set(
        edge_info["target_region_id"]
    )

    print("In node_info but not in edge_info:", len(node_ids - info_ids))
    print("In edge_info but not in node_info:", len(info_ids - node_ids))

    node_ids = set(node_info["region_id"])
    info_ids = set(edge_info["source_region_id"]) | set(
        edge_info["target_region_id"]
    )

    missing_regions = info_ids - node_ids
    print(missing_regions)

    #########################################################################

    edge_info.to_csv(DATA_FOLDER / "edge_info.csv", index=False)

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
