from pathlib import Path

import pandas as pd
import pytest

from brainglobe_data_api_connectivity.io.excel import (
    cell_reference_to_indices,
    column_reference_to_index,
    get_col_values,
    get_row_values,
    normalise_index_range,
    validate_cell_range,
    validate_cell_reference,
)


@pytest.mark.parametrize(
    ["ref", "error", "error_message"],
    [
        pytest.param("A1", None, None, id="A1"),
        pytest.param("AA10", None, None, id="AA10"),
        pytest.param("Z999", None, None, id="Z999"),
        pytest.param("", ValueError, r"Invalid cell reference: ", id="empty"),
        pytest.param(
            "123", ValueError, r"Invalid cell reference: 123", id="digits only"
        ),
        pytest.param(
            "ABC",
            ValueError,
            r"Invalid cell reference: ABC",
            id="letters only",
        ),
        pytest.param(
            "10B",
            ValueError,
            r"Invalid cell reference: 10B",
            id="starts with a digit",
        ),
        pytest.param(
            "A0A",
            ValueError,
            r"Invalid cell reference: A0A",
            id="letter digit letter",
        ),
    ],
)
def test_validate_cell_reference(ref, error, error_message):
    """Test that correct validation of (in)valid cell references."""
    if error is None:
        validate_cell_reference(ref)
    else:
        with pytest.raises(error, match=error_message):
            validate_cell_reference(ref)


@pytest.mark.parametrize(
    ["cell_reference", "expected_index"],
    [
        ("A", 0),
        ("Z", 25),
        ("AA", 26),
        ("AB", 27),
        ("AZ", 51),
        ("BA", 52),
        ("ZZ", 701),
    ],
)
def test_column_reference_to_index_valid(cell_reference, expected_index):
    """Test correct conversion of Excel column label to index."""
    assert column_reference_to_index(cell_reference) == expected_index


@pytest.mark.parametrize(
    ["ref", "expected_split_ref"],
    [
        pytest.param("A1", (0, 1), id="A1"),
        pytest.param("B10", (1, 10), id="B10"),
        pytest.param("AA500", (26, 500), id="AA500"),
        pytest.param("ZZ99", (701, 99), id="ZZ99"),
    ],
)
def test_cell_reference_to_indices(ref, expected_split_ref):
    """Test correct splitting of valid cell references."""
    assert cell_reference_to_indices(ref) == expected_split_ref


@pytest.mark.parametrize(
    ["cell_range", "error", "match"],
    [
        pytest.param(("A1", "B2"), None, None, id="valid"),
        pytest.param(
            ("A1",),
            ValueError,
            r"Cell range must contain two cell references",
            id="invalid (1 cell ref)",
        ),
        pytest.param(
            ("A1", "B2", "C3"),
            ValueError,
            r"Cell range must contain two cell references",
            id="invalid (3 cell refs)",
        ),
    ],
)
def test_validate_cell_range(cell_range, error, match):
    """Test validation of whether two references are passed."""
    if error is None:
        validate_cell_range(cell_range)
    else:
        with pytest.raises(error, match=match):
            validate_cell_range(cell_range)


@pytest.fixture
def excel_test_connectivity_matrix(tmp_path: Path):
    """Simple connectivity matrix for use in testing excel conversion."""

    df = pd.DataFrame(
        [
            ["area_1", 0, 1, 2, 3],
            ["area_2", 1, 0, 2, 3],
            ["area_3", 1, 2, 0, 3],
            ["area_4", 9, 9, 9, 0],
        ],
        columns=["", "area_1", "area_2", "area_3", "area_4"],
    )

    file_path = tmp_path / "test_connectivity_matrix.xlsx"
    df.to_excel(file_path, sheet_name="Sheet1", index=False, header=True)
    return file_path


@pytest.mark.parametrize(
    ["data_range", "col_label", "sheet", "expected"],
    [
        pytest.param(
            ("B2", "E5"),
            "A",
            "Sheet1",
            ["area_1", "area_2", "area_3", "area_4"],
            id="row labels col A",
        ),
        pytest.param(
            ("B2", "E5"),
            "B",
            None,
            [0, 1, 1, 9],
            id="values col B (sheet = None)",
        ),
    ],
)
def test_get_row_values(
    excel_test_connectivity_matrix, data_range, col_label, sheet, expected
):
    """Test getting row values matching data range."""
    result = get_row_values(
        excel_test_connectivity_matrix,
        data_range,
        col_label,
        sheet=sheet,
    )
    assert result == expected


@pytest.mark.parametrize(
    ["data_range", "col_label", "sheet", "expected"],
    [
        pytest.param(
            ("B2", "E5"),
            1,
            "Sheet1",
            ["area_1", "area_2", "area_3", "area_4"],
            id="col labels first row",
        ),
        pytest.param(
            ("B2", "E5"),
            2,
            None,
            [0, 1, 2, 3],
            id="values second row (sheet = None)",
        ),
    ],
)
def test_get_col_values(
    excel_test_connectivity_matrix, data_range, col_label, sheet, expected
):
    """Test getting col values matching data range."""
    result = get_col_values(
        excel_test_connectivity_matrix,
        data_range,
        col_label,
        sheet=sheet,
    )
    assert result == expected


@pytest.mark.parametrize(
    ["start", "end", "expected"],
    [
        pytest.param(
            (3, 1), (1, 3), ((1, 1), (3, 3)), id="top-right, bottom-left"
        ),
        pytest.param(
            (1, 3), (3, 1), ((1, 1), (3, 3)), id="bottom-left, top-right"
        ),
        pytest.param(
            (3, 3), (1, 1), ((1, 1), (3, 3)), id="bottom-right, top-left"
        ),
        pytest.param(
            (1, 1),
            (3, 3),
            ((1, 1), (3, 3)),
            id="already normal",
        ),
    ],
)
def test_normalise_index_range(start, end, expected):
    """Indices shouldalways be returned in top‑left to bottom‑right order."""
    assert normalise_index_range(start, end) == expected
