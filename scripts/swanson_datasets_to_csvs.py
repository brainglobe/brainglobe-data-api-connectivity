import os

import pandas as pd


def get_cell_range(
    cell_range: tuple[str, str],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Convert two Excel A1 references into (col_range, row_range),
    where each range is a (start, end) pair of zero‑based indices.
    """
    if len(cell_range) != 2:
        raise ValueError(
            "cell_range must contain exactly two cell references."
        )

    start_col, start_row = split_cell_reference(cell_range[0])
    end_col, end_row = split_cell_reference(cell_range[1])

    return (start_col, end_col), (start_row, end_row)


def split_cell_reference(ref):
    """
    Split an Excel A1-style cell reference into (column_label, row_number).
    Example: 'AA841' → (26, 841)
    """
    col_ref = "".join(filter(str.isalpha, ref))
    row_ref = "".join(filter(str.isdigit, ref))

    if not col_ref or not row_ref:
        raise ValueError(f"Invalid cell reference: {ref}")

    return column_reference_to_index(col_ref), int(row_ref)


def column_reference_to_index(label):
    """
    Convert Excel column labels to a 0-based index.
    A=0, B=1, ..., Z=25, AA=26, AB=27, ..., AZ=51, BA=52, ..., ZZZ=18277
    """
    label = label.upper()

    if not label.isalpha():
        raise ValueError(f"Invalid column label: {label}")

    index = 0
    for char in label:
        index = index * 26 + (ord(char) - ord("A") + 1)

    return index - 1


def get_matrix_from_excel(file, sheet_name, data_range):
    """Return DataFrame sliced to given row/column ranges."""
    col_range, row_range = get_cell_range(data_range)

    start_row, end_row = row_range
    start_col, end_col = col_range

    skiprows = start_row - 1
    nrows = end_row - start_row + 1
    usecols = list(range(start_col, end_col + 1))

    col_range, row_range = get_cell_range(data_range)

    df = pd.read_excel(
        file,
        sheet_name=sheet_name,
        skiprows=skiprows,
        nrows=nrows,
        usecols=usecols,
        header=None,
    )

    if df.shape[0] != df.shape[1]:
        raise ValueError(
            "Matrix should have equal number of rows and columns."
        )

    return df


def check_ids(file_path, sheet_name, data_range, row_with_ids, col_with_ids):
    """Specific to this dataset, checks that row and col ids match and that
    they are a range from 0 to n - 1."""
    row_ids = get_row_ids(file_path, sheet_name, data_range, row_with_ids)
    col_ids = get_col_ids(file_path, sheet_name, data_range, col_with_ids)
    if row_ids != col_ids:
        raise ValueError("Row and column IDs do not match.")

    expected_ids = list(range(1, len(row_ids) + 1))
    if row_ids != expected_ids:
        raise ValueError(
            f"Row IDs {row_ids} do not match expected {expected_ids}."
        )
    return row_ids


def validate_matrix(df, row_ids, col_ids):
    expected_rows = len(row_ids)
    expected_cols = len(col_ids)
    if df.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Matrix shape {df.shape} does not match expected "
            f"({expected_rows}, {expected_cols})."
        )


def get_row_ids(file, sheet, data_range, col_label):
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


def get_col_ids(file, sheet, data_range, row_num):
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


if __name__ == "__main__":
    data_folder = "data"
    matrices = "swansonDatasetS3 CNS data matrices JHr1.xlsx"
    matrix_sheets = [f"CNS2{sex} modules (raw)" for sex in ["m", "f"]]
    upper_left = "T8"
    lower_right = "AFU841"
    data_range = (upper_left, lower_right)
    file_path = os.path.join(data_folder, matrices)

    for sheet in matrix_sheets:
        if sheet not in pd.ExcelFile(file_path).sheet_names:
            raise ValueError(f"Sheet {sheet} not found in {file_path}.")
        m = get_matrix_from_excel(
            file_path, sheet_name=sheet, data_range=data_range
        )

        check_ids(file_path, sheet, data_range, "P", 5)

        csv_filename = sheet.replace(" ", "_") + ".csv"

        m.to_csv(
            os.path.join(data_folder, csv_filename), index=False, header=False
        )

    pass
