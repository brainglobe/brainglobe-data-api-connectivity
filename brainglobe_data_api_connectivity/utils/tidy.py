"""Tidy excel input data."""

import re
from collections.abc import Iterable


def rename_columns(columns: Iterable[str]) -> list[str]:
    """Standardise DataFrame column names.

    Returns list of strings with column names.

    Column names should never start with a digit and can only contain
     - lowercase characters a-z
     - digits 0-9
     - underscores
    """
    cleaned = []
    for column in columns:
        name = re.sub(r"[^a-z0-9]+", "_", column.lower())
        name = re.sub(r"^(\d+)([a-z0-9_]+)$", r"\2_\1", name)
        cleaned.append(re.sub(r"_+", "_", name).strip("_"))
    return cleaned
