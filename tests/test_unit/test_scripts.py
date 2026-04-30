import pytest

from brainglobe_data_api_connectivity.io.excel import (
    column_reference_to_index,
    get_cell_range,
    split_cell_reference,
)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("A", 0),
        ("Z", 25),
        ("AA", 26),
        ("AZ", 51),
    ],
)
def test_column_reference_to_index(label, expected):
    """Test basic column label conversions."""
    assert column_reference_to_index(label) == expected


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("A1", (0, 1)),
        ("C7", (2, 7)),
        ("AFU10", (column_reference_to_index("AFU"), 10)),
    ],
)
def test_split_cell_reference(ref, expected):
    """Test splitting A1 references."""
    assert split_cell_reference(ref) == expected


@pytest.mark.parametrize(
    "cell_range, expected",
    [
        (("A1", "B2"), ((0, 1), (1, 2))),
        (("ADS8", "AFU841"), ((798, 852), (8, 841))),
        (("AA1", "AA1"), ((26, 26), (1, 1))),
        (("AA1", "AA10"), ((26, 26), (1, 10))),
    ],
)
def test_get_cell_range(cell_range, expected):
    """Test converting two A1 refs into col/row ranges."""
    assert get_cell_range(cell_range) == expected


def test_invalid_reference():
    """Test invalid references raise errors."""
    with pytest.raises(ValueError):
        split_cell_reference("123")

    with pytest.raises(ValueError):
        get_cell_range(("A1",))
