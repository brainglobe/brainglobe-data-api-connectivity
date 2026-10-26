import pytest

from brainglobe_data_api_connectivity.io.loaders import (
    GIN_REPO,
    load_data,
    load_test_data,
)


@pytest.fixture
def mock_cache(tmp_path, mocker):
    """Use a separate temporary cache for each test.

    This prevents tests from affecting each other's cached files.
    """
    mocker.patch.object(GIN_REPO, "path", tmp_path)
    return tmp_path


@pytest.mark.parametrize("sex", ["invalid", "F", "Female", "M", "Male", ""])
def test_load_data_invalid_sex(sex):
    """Reject unsupported sex values before attempting a download."""
    with pytest.raises(ValueError, match="sex must be"):
        load_data(sex=sex)


def test_load_refresh(mock_cache):
    """Check that refresh replaces a modified cached file."""
    load_test_data()

    cached_file = mock_cache / "test-data/small-nodes.csv"
    original_content = cached_file.read_text()

    # Modify the cached file.
    modified_content = original_content.replace("alpha", "modified_alpha")
    assert modified_content != original_content
    cached_file.write_text(modified_content)

    # Without refresh, the modified file stays cached.
    load_test_data(refresh=False)
    assert cached_file.read_text() == modified_content

    # With refresh, the original file is downloaded again.
    load_test_data(refresh=True)
    assert cached_file.read_text() == original_content
