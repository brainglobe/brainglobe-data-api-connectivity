import pytest

from brainglobe_data_api_connectivity.io.excel import (
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
            "123", ValueError, r"Invalid cell reference: 123", id="digits_only"
        ),
        pytest.param(
            "ABC",
            ValueError,
            r"Invalid cell reference: ABC",
            id="letters only",
        ),
        pytest.param(
            "1",
            ValueError,
            r"Invalid cell reference: 1",
            id="digits only",
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
