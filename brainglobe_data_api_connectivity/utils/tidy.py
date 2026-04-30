import os

import pandas as pd


def rename_columns(columns: pd.Index) -> pd.Index:
    return (
        columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("\n", "_")
        .str.replace(".", "")
        .str.replace("#", "")
    )


def consolidate_info_files(data_folder: str) -> None:
    """Load all info files and check whether they are the same.
    Remove duplicates and rename the first one to simply info.csv.
    """

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
