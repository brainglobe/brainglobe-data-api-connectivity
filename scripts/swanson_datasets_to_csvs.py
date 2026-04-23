import pandas as pd
import os


def get_cell_range(cell_range: tuple[str, str]) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Convert two Excel A1 references into (col_range, row_range),
    where each range is a (start, end) pair of zero‑based indices.
    """
    if len(cell_range) != 2:
        raise ValueError("cell_range must contain exactly two cell references.")

    start_col, start_row = split_cell_reference(cell_range[0])
    end_col, end_row = split_cell_reference(cell_range[1])

    return (start_col, end_col), (start_row, end_row)


def split_cell_reference(ref):
    """
    Split an Excel A1-style cell reference into (column_label, row_number).
    Example: 'AA841' → (26, 841)
    """
    col_ref = ''.join(filter(str.isalpha, ref))
    row_ref = ''.join(filter(str.isdigit, ref))

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
        index = index * 26 + (ord(char) - ord('A') + 1)

    return index - 1


def get_matrix_from_excel(file, sheet_name, row_range, col_range):
    """Return DataFrame sliced to given row/column ranges."""
    start_row, end_row = row_range
    start_col, end_col = col_range

    skiprows = start_row - 1
    nrows = end_row - start_row + 1

    df = pd.read_excel(
        file,
        sheet_name=sheet_name,
        skiprows=skiprows,
        nrows=nrows,
        usecols=lambda idx: start_col <= idx <= end_col,
    )
    return df


if __name__ == "__main__":
    data_folder = "data"

    matrices = "swansonDatasetS3 CNS data matrices JHr1.xlsx"
    matrice_tabs = [f"CNS2{sex} modules (raw)" for sex in ["m", "f"]]

    row_headers = ("P8", "P841")
    column_headers = ("5T", "5AFU")
    matrix= ("ADS8", "AFU841")

    file_path = os.path.join(data_folder, matrices)

    cell_range = get_cell_range(matrix)
