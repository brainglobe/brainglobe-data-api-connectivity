"""Excel-related helper functions."""

import re
from pathlib import Path

import pandas as pd


def get_df_from_excel(
    file: Path,
    sheet_name: str,
    data_range: tuple[str, str],
    header: int | None = None,
) -> pd.DataFrame:
    """Return DataFrame sliced to given row/column ranges."""
    col_range, row_range = get_cell_range(data_range)

    start_row, end_row = row_range
    start_col, end_col = col_range

    skiprows = start_row - 1
    nrows = end_row - start_row + 1
    usecols = list(range(start_col, end_col + 1))

    df = pd.read_excel(
        file,
        sheet_name=sheet_name,
        skiprows=skiprows,
        nrows=nrows,
        usecols=usecols,
        header=header,
    )
    return df


def validate_cell_reference(ref: str) -> None:
    """Validate that a cell reference is letters A–Z followed by digits 0–9."""

    pattern = r"^[A-Za-z]+[0-9]+$"

    if not re.match(pattern, ref):
        raise ValueError(f"Invalid cell reference: {ref}")


def cell_reference_to_indices(ref: str) -> tuple[int, int]:
    """Split a cell reference into zero-based column index and row number."""
    validate_cell_reference(ref)
    col_ref = "".join(filter(str.isalpha, ref))
    row_ref = "".join(filter(str.isdigit, ref))
    return column_reference_to_index(col_ref), int(row_ref)


def column_reference_to_index(label: str) -> int:
    """converts Excel column labels into numeric indices."""
    label = label.upper()

    index = 0
    for char in label:
        index = index * 26 + (ord(char) - ord("A") + 1)

    return index - 1


def normalise_index_range(
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Normalise two (col, row) index pairs to top-left → bottom-right.

    Examples:
        ((3, 1), (1, 3)) becomes ((1, 1), (3, 3))
        ((1, 3), (3, 1)) becomes ((1, 1), (3, 3))
        ((3, 3), (1, 1)) becomes ((1, 1), (3, 3))
        ((1, 1), (3, 3)) stays ((1, 1), (3, 3))
    """
    (col_a, row_a), (col_b, row_b) = start, end

    min_col = min(col_a, col_b)
    max_col = max(col_a, col_b)
    min_row = min(row_a, row_b)
    max_row = max(row_a, row_b)

    return (min_col, min_row), (max_col, max_row)


def validate_cell_range(cell_range: tuple[str, str]) -> None:
    """Validate that cell_range contains two cell references."""
    if len(cell_range) != 2:
        raise ValueError("Cell range must contain two cell references.")


def get_cell_range(
    cell_range: tuple[str, str],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert two cell references into column and row index ranges."""
    validate_cell_range(cell_range)

    start = cell_reference_to_indices(cell_range[0])
    end = cell_reference_to_indices(cell_range[1])

    (start_col, start_row), (end_col, end_row) = normalise_index_range(
        start, end
    )

    return (start_col, end_col), (start_row, end_row)


def get_row_values(
    file: Path,
    data_range: tuple[str, str],
    col_label: str,
    sheet: str | int | None = None,
) -> list[int]:
    """Return values from a row range.

    Args
        file: Path to the Excel file.
        sheet: Sheet name or index. If None, the first sheet is used.
        data_range: tuple of cell references defining the data range,
            e.g. ("A1", "C5").
        col_label: Excel column label (e.g. "A", "B", "AA") to extract values
        from.

    """
    if sheet is None:
        sheet = 0

    (_, _), (r0, r1) = get_cell_range(data_range)
    col = column_reference_to_index(col_label)

    return (
        pd.read_excel(
            file,
            sheet_name=sheet,
            skiprows=r0 - 1,
            nrows=r1 - r0 + 1,
            usecols=[col],
            header=None,
        )
        .iloc[:, 0]
        .tolist()
    )


def get_col_values(
    file: Path,
    data_range: tuple[str, str],
    row_num: int,
    sheet: str | int | None = None,
) -> list[int]:
    """Return values from a row range.

    Args
        file: Path to the Excel file.
        sheet: Sheet name or index. If None, the first sheet is used.
        data_range: tuple of cell references defining the data range,
            e.g. ("A1", "C5").
        col_label: Excel row label (e.g. 1, 2) to extract values from.

    """
    if sheet is None:
        sheet = 0
    (c0, c1), (_, _) = get_cell_range(data_range)
    return (
        pd.read_excel(
            file,
            sheet_name=sheet,
            skiprows=row_num - 1,
            nrows=1,
            usecols=list(range(c0, c1 + 1)),
            header=None,
        )
        .iloc[0]
        .tolist()
    )
