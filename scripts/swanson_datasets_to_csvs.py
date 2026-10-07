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

import polars as pl

from brainglobe_data_api_connectivity.connections import Connections
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


DATA_FOLDER = Path("data")
MATRIX_IDS = ["CNS2f", "CNS2m"]

EDGE_INFO_MORPH_DICT = {"side": {"one": 1, "two": 2, "left": 1, "right": 2}}

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


if __name__ == "__main__":
    # Clean and save edge_information CSV
    edge_info = excel.get_df_from_excel(
        SWANSON_PARAMS["edge_info_file"], header=[0, 1]
    )
    edge_info.columns = tidy.rename_columns(edge_info.columns)
    edge_info = edge_info.with_columns(pl.col(pl.String).replace("", None))

    side_columns = [col for col in edge_info.columns if "side" in col]
    edge_info = edge_info.with_columns(
        pl.col(side_columns).replace_strict(
            EDGE_INFO_MORPH_DICT["side"],
            default=None,
            return_dtype=pl.Int64,
        )
    )

    edge_info = edge_info.with_columns(
        pl.col("connection_reported_value").str.to_lowercase(),
    )
    for region in ["origin", "termination"]:
        edge_info = edge_info.with_columns(
            pl.concat_str(
                f"connection_{region}_region_abbr",
                f"connection_{region}_region_side",
                separator="_",
            ).alias(f"{region}_region_id")
        )

    # Process each matrix sheet
    for matrix_id in MATRIX_IDS:
        sheet = next(
            sheet
            for sheet in SWANSON_PARAMS["matrix_sheets"]
            if matrix_id in sheet
        )

        # Load and tidy node info
        node_info = excel.get_df_from_excel(
            SWANSON_PARAMS["matrix_file"],
            sheet_name=sheet,
            data_range=SWANSON_PARAMS["node_info_range"],
            header=0,
        )
        node_info = node_info.rename({node_info.columns[0]: "Side"})
        node_info.columns = tidy.rename_columns(node_info.columns)

        node_info = node_info.with_columns(
            pl.col("side", "level_1", "level_2", "level_3", "mrcc").cast(
                pl.Int64
            )
        )
        node_info = (
            node_info.with_columns(
                region_id=pl.concat_str("abbr", "side", separator="_")
            )
            .with_columns(pl.col(pl.String).replace("•", "0"))
            .with_row_index("region_idx")
        )

        # Load and validate matrix and ids
        processed_matrix = excel.get_df_from_excel(
            SWANSON_PARAMS["matrix_file"],
            sheet,
            SWANSON_PARAMS["matrix_range"],
        )
        validate_input.validate_adjacency_matrix(processed_matrix)

        edge_table = convert.convert_matrix_to_edge_table(
            processed_matrix,
            region_ids=node_info["region_idx"],
        ).with_columns(pl.col("weight").cast(pl.Int64))

        # Save outputs
        output_folder = DATA_FOLDER / matrix_id
        output_folder.mkdir(exist_ok=True)

        edge_table.write_csv(
            output_folder / f"{matrix_id}_edge_table.csv",
            include_header=False,
        )
        node_info.write_csv(output_folder / f"{matrix_id}_node_info.csv")

        print(
            f"Saved node info and edge table for {matrix_id} "
            f"in {output_folder}"
        )

    # Map region IDs in edge_info to node indices
    region_index = dict(zip(node_info["region_id"], node_info["region_idx"]))

    for region in ["origin", "termination"]:
        edge_info = edge_info.with_columns(
            pl.col(f"{region}_region_id")
            .replace_strict(
                region_index,
                default=None,
                return_dtype=pl.Int64,
            )
            .alias(f"{region}_region_idx")
        )

    edge_info.write_csv(DATA_FOLDER / "edge_info.csv")
    print(f"Saved edge_info.csv in {DATA_FOLDER}")

    for matrix_id in MATRIX_IDS:
        output_folder = DATA_FOLDER / matrix_id
        sex = "male" if matrix_id == "CNS2m" else "female"

        edge_info.filter(pl.col("male_or_female") == sex).write_csv(
            output_folder / f"{matrix_id}_edge_info.csv"
        )
        print(f"Saved {matrix_id}_edge_info.csv in {output_folder}")

    # Check that the generated CSVs can be loaded into Connections
    print(
        "Checking whether CSV files can be used to create Connections objects"
    )

    for matrix_id in MATRIX_IDS:
        output_folder = DATA_FOLDER / matrix_id
        connections = Connections.from_files(
            node_info=output_folder / f"{matrix_id}_node_info.csv",
            edge_table=output_folder / f"{matrix_id}_edge_table.csv",
            edge_info=output_folder / f"{matrix_id}_edge_info.csv",
            edge_info_from_col="origin_region_idx",
            edge_info_to_col="termination_region_idx",
            node_index_column="region_idx",
        )

        print(f"Successfully created Connections for {matrix_id}")
        print(f"Number of nodes: {connections.network.num_nodes()}")
        print(f"Number of edges: {connections.network.num_edges()}")
