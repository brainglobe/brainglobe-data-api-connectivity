from pathlib import Path

import numpy as np
import pandas as pd

from brainglobe_data_api_connectivity.io.excel import (
    get_col_ids,
    get_df_from_excel,
    get_row_ids,
)
from brainglobe_data_api_connectivity.io.validate_input import (
    check_ids,
    validate_matrix,
)
from brainglobe_data_api_connectivity.utils.convert import (
    convert_matrix_to_edge_table,
    lookup_node_index,
)
from brainglobe_data_api_connectivity.utils.tidy import (
    consolidate_info_files,
    rename_columns,
)

morph_dict = {"one": 1, "two": 2}


def _morph(region_side: str) -> int:
    return morph_dict[region_side]


if __name__ == "__main__":
    data_folder = "data"
    matrices = "swansonDatasetS3 CNS data matrices JHr1.xlsx"
    matrix_sheets = [f"CNS2{sex} modules" for sex in ["m", "f"]]
    data_range = ("T8", "AFU841")
    info_range = ("A7", "S841")

    file_path = Path(data_folder) / matrices

    metadata = "swansonDatasetS2 CNS CRs JHr1.xlsx"
    metadata_file = Path(data_folder) / metadata

    # df = pd.read_excel(metadata_file)
    # df.to_csv(os.path.join(data_folder, "edge_metadata.csv"), index=False)

    for sheet in matrix_sheets:
        if sheet not in pd.ExcelFile(file_path).sheet_names:
            raise ValueError(f"Sheet {sheet} not found in {file_path}.")
        matrix = get_df_from_excel(
            file_path, sheet_name=sheet, data_range=data_range
        )

        row_ids = get_row_ids(file_path, sheet, data_range, "P")
        col_ids = get_col_ids(file_path, sheet, data_range, 5)
        check_ids(row_ids, col_ids)
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
            Path(data_folder) / f"{sheet.replace(' ', '_')}_edge_table.csv",
            edge_table,
            fmt="%d",
            delimiter=",",
        )

        info.to_csv(
            Path(data_folder) / f"{sheet.replace(' ', '_')}_info.csv",
            index=False,
        )

    consolidate_info_files(data_folder)

    pass
