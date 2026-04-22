from pathlib import Path

import pytest


@pytest.fixture
def DATA_DIR() -> Path:
    """Path to static directory holding test data."""
    return Path(__file__).parent / ".." / "data"
