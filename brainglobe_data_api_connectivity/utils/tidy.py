"""Tidy excel input data"""

import pandas as pd


def rename_columns(columns: pd.Index) -> pd.Index:
    """Standardise DataFrame column names.

    Returns a new index for the columns.

    Column names should never start with a diget and can only contain
     - lowercase characters a-z
     - digits 0-9
     - underscores
    """
    cleaned = (
        columns.str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.replace(
            r"^(\d+)([a-z0-9_]+)$", r"\2_\1", regex=True
        )  # move leading digits to the end
        .str.replace(r"_+", "_", regex=True)  # collapse multiple _
        .str.strip("_")  # trim any underscores from start or end
    )
    return pd.Index(cleaned)
