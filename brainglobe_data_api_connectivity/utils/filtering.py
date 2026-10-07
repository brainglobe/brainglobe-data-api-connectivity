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
        if value not in available_values:
            raise ValueError(
                f"Unknown filter value {value!r} for column {column!r}. "
                f"Available values: {available_values}"
            )


def data_filter(
    data: pl.DataFrame,
    filters: dict[str, Any] | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Return rows matching filters, then select columns in requested order.

    By default, all rows and columns are preserved. Passing `None` for filters
    and columns preserves all rows and columns respectively.

    Unknown column names or unavailable filter values raise `ValueError`.
    """
    if filters:
        validate_filters(data, filters)
        data = data.filter(**filters)
    if columns is not None:
        validate_columns(data, columns)
        data = data.select(columns)
    return data
