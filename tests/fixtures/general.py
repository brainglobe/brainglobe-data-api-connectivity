import re
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def DATA_DIR() -> Path:
    """Path to static directory holding test data."""
    return Path(__file__).parent / ".." / "data"


@pytest.fixture
def raises_error() -> Callable[[Exception], pytest.RaisesExc]:
    """Return a `pytest.raises` context that seeks the given Exception."""

    def _inner(error: Exception) -> pytest.RaisesExc:
        return pytest.raises(type(error), match=re.escape(str(error)))

    return _inner
