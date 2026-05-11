import pandas as pd
import pytest

from brainglobe_data_api_connectivity.utils.tidy import (
    rename_columns,
)


@pytest.mark.parametrize(
    ["name", "expected"],
    [
        pytest.param("first name", "first_name", id="space"),
        pytest.param("Age\nWeeks", "age_weeks", id="newline"),
        pytest.param("CAPITALS", "capitals", id="capitals"),
        pytest.param(
            "#-£invalid-$%^&(chars[}*.@", "invalid_chars", id="invalid chars"
        ),
        pytest.param(
            "123leading_digits", "leading_digits_123", id="leading digits"
        ),
        pytest.param(
            "___many__underscores___", "many_underscores", id="underscores"
        ),
    ],
)
def test_rename_columns(name, expected):
    """Test whether rename_columns works as expected."""
    renamed = rename_columns(pd.Index([name]))[0]
    assert renamed == expected
