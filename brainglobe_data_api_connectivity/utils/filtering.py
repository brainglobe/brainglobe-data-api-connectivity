from math import isnan
from typing import Any, Iterable

import polars as pl


def filter_options(data: pl.DataFrame) -> dict[str, list]:
    """Return unique values for each column in ascending order."""
    return {
        column: data[column].unique().sort().to_list()
        for column in data.columns
    }


def validate_columns(data: pl.DataFrame, columns: Iterable[str]) -> None:
    """Validate requested names against the available columns."""
    invalid_columns = [
        column for column in columns if column not in data.columns
    ]
    if invalid_columns:
        raise ValueError(
            f"Unknown columns: {invalid_columns}. "
            f"Available columns: {data.columns}"
        )


def validate_filters(data: pl.DataFrame, filters: dict[str, Any]) -> None:
    """Validate filter names and values against the available options."""
    validate_columns(data, filters)
    for column, value in filters.items():
        available_values = data[column].unique(maintain_order=True).to_list()
        if value not in available_values and not (
            isinstance(value, float)
            and isnan(value)
            and any(
                isinstance(option, float) and isnan(option)
                for option in available_values
            )
        ):
            raise ValueError(
                f"Unknown filter value {value!r} for column {column!r}. "
                f"Available values: {available_values}"
            )


def data_filter(
    data: pl.DataFrame,
    filters: dict[str, Any] | None = None,
    columns: list[str] | None = None,
    validation_data: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Filter rows and select columns from a DataFrame.

    Filters and columns are validated against `validation_data` if provided,
    otherwise against `data`. Invalid column names or filter values raise
    `ValueError`.

    A filter value of `None` matches null values. If no filters or columns
    are specified, all rows or columns are preserved, respectively.
    """
    validation_data = data if validation_data is None else validation_data
    if filters:
        validate_filters(validation_data, filters)
        data = data.filter(
            pl.col(column).is_null()
            if value is None
            else pl.col(column) == value
            for column, value in filters.items()
        )
    if columns is not None:
        validate_columns(validation_data, columns)
        data = data.select(columns)
    return data
