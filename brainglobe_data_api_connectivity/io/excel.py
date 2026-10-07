"""Excel-related helper functions."""

import re
from pathlib import Path

import polars as pl


def get_df_from_excel(
    file: Path,
    sheet_name: str | None = None,
    data_range: tuple[str, str] | None = None,
    header: int | list[int] | None = None,
) -> pl.DataFrame:
    """Return DataFrame sliced to given row/column ranges."""
    skiprows, nrows, usecols = 0, None, None
    if data_range is not None:
        (start_col, end_col), (start_row, end_row) = get_cell_range(data_range)
        skiprows = start_row - 1
        nrows = end_row - start_row + 1
        usecols = list(range(start_col, end_col + 1))

    names = None
    consumed_rows = 0
    if isinstance(header, list):
        # Pandas supports header=[0, 1] as MultiIndex columns; Polars has no
        # MultiIndex. So the header rows are read separately and combined
        # into flat names.
        consumed_rows = max(header) + 1
        headings = pl.read_excel(
            file,
            sheet_name=sheet_name,
            columns=usecols,
            has_header=False,
            drop_empty_rows=False,
            drop_empty_cols=False,
            read_options={"skip_rows": skiprows, "n_rows": consumed_rows},
        )
        rows = []
        for i in header:
            row = headings.row(i)
            if i != header[-1]:
                row = pl.Series(row).fill_null(strategy="forward").to_list()
            rows.append(row)
        names = [
            "_".join(str(v) for v in values if v is not None)
            for values in zip(*rows)
        ]
        header = None
    elif header is not None:
        consumed_rows = header + 1

    df = pl.read_excel(
        file,
        sheet_name=sheet_name,
        columns=usecols,
        has_header=header is not None,
        drop_empty_rows=False,
        drop_empty_cols=False,
        infer_schema_length=None,
        read_options={
            "header_row": skiprows + header if header is not None else None,
            "skip_rows": 0 if header is not None else skiprows + consumed_rows,
            "n_rows": nrows - consumed_rows if nrows is not None else None,
        },
    )
    if names is not None or header is None:
        df.columns = names or [str(i) for i in range(df.width)]
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
