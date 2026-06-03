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
    matrix_dict = {}
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
        edge_table_processed_for_check = convert.convert_matrix_to_edge_table(
            processed_matrix, include_zeros=True
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

        # TODO: QUESTION FEMALE MALE? WHAT DOES GET INCLUDED?
        if matrix_id == "CNS2m":
            edge_info_sheet = edge_info[edge_info["male_or_female"] == "male"]
        elif matrix_id == "CNS2f":
            edge_info_sheet = edge_info[
                edge_info["male_or_female"] == "female"
            ]

        edge_info_sheet.to_csv(
            output_folder / f"{matrix_id}_edge_info.csv", index=False
        )

        # Add edge_id to edge_table_raw
        edge_id = (
            node_info["region_id"].iloc[edge_table_raw[:, 0]].values
            + "-"
            + node_info["region_id"].iloc[edge_table_raw[:, 1]].values
        )

        processed_strength = edge_table_processed_for_check[:, 2]
        edge_table_df = pd.DataFrame(
            {
                "source_region_idx": edge_table_raw[:, 0],
                "target_region_idx": edge_table_raw[:, 1],
                "raw_connection_strength": edge_table_raw[:, 2],
                "processed_connection_strength": processed_strength,
                "edge_id": edge_id,
            }
        )
        # also save node_info and edge_table in dict
        matrix_dict[sheet] = {
            "edge_table": edge_table_df,
            "node_info": node_info,
            "edge_info": edge_info_sheet,
        }

    #########################################################################
    #######################       CHECKS           ##########################
    #########################################################################

    # TODO investigate whether ids are as expected and why merged
    #  is not as expected

    node_region_ids = set(node_info["region_id"])
    edge_region_ids = set(edge_info_sheet["source_region_id"]) | set(
        edge_info_sheet["target_region_id"]
    )

    print(
        "In node_info but not in edge_info:",
        len(node_region_ids - edge_region_ids),
    )
    print(
        "In edge_info but not in node_info:",
        len(edge_region_ids - node_region_ids),
    )

    missing_regions = edge_region_ids - node_region_ids
    print("Missing regions: ", missing_regions)

    # check whether edge info male_or_female column
    unique_m_f_labels = set(edge_info["male_or_female"])
    print("male_or_female contains the following labels: ", unique_m_f_labels)

    F, M = matrix_dict["CNS2f modules"], matrix_dict["CNS2m modules"]
    for S, sex in zip([F, M], ["female", "male"]):
        print("----------------------------------------------------------")
        print(f"{sex.upper()}")
        print("----------------------------------------------------------")
        unique_values = set(
            S["edge_info"]["connection_reported_value"].str.lower()
        )
        print(f"{sex} edge_info unique labels:", len(unique_values))
        print(
            f"Missing: {
                set(list(RAW_CONNECTION_STRENGTH.keys())) - unique_values
            }"
        )

        edges1 = set(S["edge_info"]["edge_id"])
        edges2 = set(S["edge_table"]["edge_id"])

        only_in_edge_info = edges1 - edges2
        only_in_edge_table = edges2 - edges1

        print("")
        print("Number of edges")
        print(f"in edge_info but NOT in edge_table: {len(only_in_edge_info)}")
        print(f"in edge_table but NOT in edge_info: {len(only_in_edge_table)}")
        print("")

        # Merge the two sources on edge_id
        edge_table_and_info = S["edge_info"].merge(
            S["edge_table"],
            on="edge_id",
            suffixes=("_edge_info", "_raw_matrix_df"),
        )

        # Find mismatches between
        mismatches = edge_table_and_info[
            edge_table_and_info["raw_connection_strength_edge_info"]
            != edge_table_and_info["raw_connection_strength_raw_matrix_df"]
        ]

        if mismatches.empty:
            print("All raw_connection_strength values match per edge_id.")
        else:
            print("Mismatches found:")
            print(mismatches)

        ###
        print("")
        print("MISMATCH 0:")
        print(mismatches.iloc[0])
        ###
        print("MISMATCH 100:")
        print(mismatches.iloc[100])
        ###
        print("MISMATCH 1000:")
        print(mismatches.iloc[1000])
        ###

        # find edge_id duplicates in edge_table_and_info
        multiple_edges = edge_table_and_info[
            edge_table_and_info.duplicated("edge_id", keep=False)
        ].sort_values("edge_id")

        # select those duplicate edges that have different
        # "connection_reported_value"
        multiple_edges_with_diff_values = (
            multiple_edges.groupby("edge_id")
            .filter(lambda g: g["connection_reported_value"].nunique() > 1)
            .sort_values("edge_id")
        )

    #########################################################################

    # Are there no mismatches when selecting max processed value?
    ONLY_MAX_RAW_VALUE_FROM_EDGE_INFO = True
    for S, sex in zip([F, M], ["female", "male"]):
        # get index of the max processed_connection_strength per edge_id
        edge_info_selection = edge_info.sort_values("edge_id")
        if ONLY_MAX_RAW_VALUE_FROM_EDGE_INFO:
            idx = (
                S["edge_info"]
                .groupby("edge_id")["raw_connection_strength"]
                .idxmax()
            )

            # slice edge_info to get the full rows
            max_edge_info = S["edge_info"].loc[idx]

            # rename column for clarity
            max_edge_info = max_edge_info.rename(
                columns={
                    "raw_connection_strength": "max_raw_connection_strength"
                }
            )

            edge_info_selection = max_edge_info

        left = edge_info_selection.rename(
            columns={
                c: f"{c}_edge_info"
                for c in edge_info_selection.columns
                if c != "edge_id"
            }
        )

        right = S["edge_table"].rename(
            columns={
                c: f"{c}_matrix"
                for c in S["edge_table"].columns
                if c != "edge_id"
            }
        )

        merged = left.merge(right, on="edge_id", how="left")

        # mismatches
        if ONLY_MAX_RAW_VALUE_FROM_EDGE_INFO:
            mismatches = merged[
                merged["raw_connection_strength_matrix"]
                != merged["max_raw_connection_strength_edge_info"]
            ]
        else:
            mismatches = merged[
                merged["raw_connection_strength_matrix"]
                != merged["raw_connection_strength_edge_info"]
            ]

        strong_mismatches = mismatches[
            mismatches["connection_reported_value_edge_info"] == "strong"
        ]

        na_mismatches = mismatches[
            mismatches["processed_connection_strength_matrix"].isna()
        ]
        na_mismatch_counts = na_mismatches[
            "connection_reported_value_edge_info"
        ].value_counts()
        print("Counts per NaN mismatch type:")
        print(na_mismatch_counts)

        non_na_mismatches = mismatches[
            mismatches["processed_connection_strength_matrix"].notna()
        ]
        non_na_mismatch_counts = na_mismatches[
            "connection_reported_value_edge_info"
        ].value_counts()
        print("Counts per non-NaN mismatch type:")
        print(non_na_mismatch_counts)

        strong_na_mismatches = na_mismatches[
            na_mismatches["connection_reported_value_edge_info"] == "strong"
        ]

        print(strong_na_mismatches.iloc[0])

        strong_mismatches = mismatches[
            mismatches["connection_reported_value_edge_info"] == "strong"
        ]

        print(strong_mismatches.iloc[0])

        if mismatches.empty:
            print("All max raw_connection_strength values match per edge_id.")
        else:
            print(f"Mismatches found for {sex}:")
            print(mismatches)
        ###
        print("")
        print("MISMATCH 0:")
        print(mismatches.iloc[0])
        ###
        print("MISMATCH 100:")
        print(mismatches.iloc[100])
        ###
        print("MISMATCH 1000:")
        print(mismatches.iloc[1000])
        ###

        mismatches.to_csv(f"{sex}_mismatches.csv", index=False)

    pass
