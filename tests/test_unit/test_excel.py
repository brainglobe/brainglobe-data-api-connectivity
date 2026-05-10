import pytest

from brainglobe_data_api_connectivity.io.excel import (
    cell_reference_to_indices,
    column_reference_to_index,
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
            r"cell_range must contain two cell references",
            id="invalid (1 cell ref)",
        ),
        pytest.param(
            ("A1", "B2", "C3"),
            ValueError,
            r"cell_range must contain two cell references",
            id="invalid (3 cell refs",
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
