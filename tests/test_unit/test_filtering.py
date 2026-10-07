import polars as pl
import pytest
from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity.utils.filtering import (
    data_filter,
    filter_options,
    validate_columns,
    validate_filters,
)


@pytest.fixture
def data() -> pl.DataFrame:
    """Simple data unrelated to graph connections."""
    return pl.DataFrame({"name": ["B", "A", "B"], "group": ["x", "y", "x"]})


def test_filter_options(data):
    """Test filter_options gives unique options sorted in ascending order."""
    assert filter_options(data) == {"name": ["A", "B"], "group": ["x", "y"]}


@pytest.mark.parametrize(
    ("validator", "values"),
    [
        pytest.param(validate_columns, ["group", "name"], id="valid columns"),
        pytest.param(validate_columns, [], id="empty columns"),
        pytest.param(
            validate_filters, {"name": "A", "group": "y"}, id="valid filters"
        ),
        pytest.param(validate_filters, {}, id="empty filters"),
    ],
)
def test_valid_columns_and_filters(data, validator, values):
    """Existing names and values, including empty inputs, are valid."""
    validator(data, values)


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(["missing"], id="unknown name"),
        pytest.param([0], id="numeric column"),
    ],
)
def test_invalid_columns(data, columns):
    """Invalid column names report the available columns."""
    with pytest.raises(ValueError) as exc_info:
        validate_columns(data, columns)

    assert str(exc_info.value) == (
        f"Unknown columns: {columns}. Available columns: ['name', 'group']"
    )


@pytest.mark.parametrize(
    ("filters", "columns", "message"),
    [
        pytest.param(
            None,
            ["missing"],
            "Unknown columns: ['missing']. "
            "Available columns: ['name', 'group']",
            id="invalid selected column",
        ),
        pytest.param(
            {"group": "missing"},
            None,
            "Unknown filter value 'missing' for column 'group'. "
            "Available values: ['x', 'y']",
            id="invalid filter value",
        ),
    ],
)
def test_invalid_filter(data, filters, columns, message):
    """Invalid selected columns and filter values raise useful errors."""
    with pytest.raises(ValueError) as exc_info:
        data_filter(data, filters=filters, columns=columns)

    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        pytest.param(
            {"missing": "x"},
            "Unknown columns: ['missing']. "
            "Available columns: ['name', 'group']",
            id="invalid filter name (column)",
        ),
        pytest.param(
            {"group": "missing"},
            "Unknown filter value 'missing' for column 'group'. "
            "Available values: ['x', 'y']",
            id="invalid filter value",
        ),
    ],
)
def test_validate_filters_invalid(data, filters, message):
    """Invalid filter names and values."""
    with pytest.raises(ValueError) as exc_info:
        validate_filters(data, filters)

    assert str(exc_info.value) == message
    with pytest.raises(ValueError) as exc_info:
        data_filter(data, filters)

    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("filters", "columns", "expected"),
    [
        pytest.param(
            None,
            None,
            pl.DataFrame({"name": ["B", "A", "B"], "group": ["x", "y", "x"]}),
            id="None (full data returned)",
        ),
        pytest.param(
            {},
            None,
            pl.DataFrame({"name": ["B", "A", "B"], "group": ["x", "y", "x"]}),
            id="empty filter (full data returned)",
        ),
        pytest.param(
            {"name": "A"},
            None,
            pl.DataFrame({"name": ["A"], "group": ["y"]}),
            id="filter (A) only",
        ),
        pytest.param(
            None,
            ["group", "name"],
            pl.DataFrame({"group": ["x", "y", "x"], "name": ["B", "A", "B"]}),
            id="columns only (swap original order)",
        ),
        pytest.param(
            {"group": "y"},
            ["name"],
            pl.DataFrame({"name": ["A"]}),
            id="filter + column",
        ),
        pytest.param(
            {"name": "A", "group": "x"},
            ["name"],
            pl.DataFrame(schema={"name": pl.String}),
            id="valid filters with no matching combination",
        ),
    ],
)
def test_data_filter(data, filters, columns, expected):
    """Filtering and selection preserve the input data and requested order."""
    original = data.clone()

    result = data_filter(data, filters, columns)

    assert_frame_equal(result, expected)
    assert_frame_equal(data, original)
