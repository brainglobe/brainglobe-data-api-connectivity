import polars as pl
import pytest
from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity.io.excel import (
    cell_reference_to_indices,
    column_reference_to_index,
    get_df_from_excel,
    normalise_index_range,
    validate_cell_range,
    validate_cell_reference,
)


@pytest.mark.parametrize(
    ["ref", "error"],
    [
        pytest.param("A1", None, id="A1"),
        pytest.param("AA10", None, id="AA10"),
        pytest.param("Z999", None, id="Z999"),
        pytest.param("", ValueError("Invalid cell reference: "), id="empty"),
        pytest.param(
            "123", ValueError("Invalid cell reference: 123"), id="digits only"
        ),
        pytest.param(
            "ABC",
            ValueError("Invalid cell reference: ABC"),
            id="letters only",
        ),
        pytest.param(
            "10B",
            ValueError("Invalid cell reference: 10B"),
            id="starts with a digit",
        ),
        pytest.param(
            "A0A",
            ValueError("Invalid cell reference: A0A"),
            id="letter digit letter",
        ),
    ],
)
def test_validate_cell_reference(ref, error, raises_error):
    """Test that correct validation of (in)valid cell references."""
    if error is None:
        validate_cell_reference(ref)
    else:
        with raises_error(error):
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
    ["cell_range", "error"],
    [
        pytest.param(("A1", "B2"), None, id="valid"),
        pytest.param(
            ("A1",),
            ValueError("Cell range must contain two cell references"),
            id="invalid (1 cell ref)",
        ),
        pytest.param(
            ("A1", "B2", "C3"),
            ValueError("Cell range must contain two cell references"),
            id="invalid (3 cell refs)",
        ),
    ],
)
def test_validate_cell_range(cell_range, error, raises_error):
    """Test validation of whether two references are passed."""
    if error is None:
        validate_cell_range(cell_range)
    else:
        with raises_error(error):
            validate_cell_range(cell_range)


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


@pytest.mark.parametrize(
    ["filename", "sheet_name", "data_range", "header", "expected"],
    [
        pytest.param(
            "mini-nodes-matrix.xlsx",
            "combined_header",
            ("A2", "B7"),
            [0, 1],
            pl.DataFrame(
                {
                    "node_name": ["A", "B", "C", "D"],
                    "node_idx": [0, 1, 2, 3],
                }
            ),
            id="node information (combined header)",
        ),
        pytest.param(
            "mini-nodes-matrix.xlsx",
            "connectivity",
            ("F3", "I6"),
            None,
            pl.DataFrame(
                {
                    "0": [0, 0, 0, 0],
                    "1": [0.1, 0.0, 0.0, 0.0],
                    "2": [10, 1, 0, 0],
                    "3": [0, 0, 10, 0],
                }
            ),
            id="matrix without header",
        ),
        pytest.param(
            "mini-edge-info.xlsx",
            "edge_info",
            ("A2", "E7"),
            0,
            pl.DataFrame(
                {
                    "from_id": ["A", "B", "A", "A", "C"],
                    "to_id": ["B", "C", "C", "C", "D"],
                    "from": [0, 1, 0, 0, 2],
                    "to": [1, 2, 2, 2, 3],
                    "used": ["yes", "yes", "yes", "no", "yes"],
                }
            ),
            id="edge information",
        ),
    ],
)
def test_get_df_from_excel(
    DATA_DIR, filename, sheet_name, data_range, header, expected
):
    """Extract mini_G data from the saved Excel files."""
    result = get_df_from_excel(
        DATA_DIR / filename,
        sheet_name=sheet_name,
        data_range=data_range,
        header=header,
    )

    assert_frame_equal(result, expected)
