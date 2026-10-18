import pooch
import pytest
from polars.testing import assert_frame_equal

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.io.loaders import (
    gin_url,
    load_data,
    load_test_data,
)


@pytest.fixture
def mock_cache(tmp_path, mocker):
    """Use a separate temporary cache for each test.

    This prevents tests from affecting each other's cached files.
    """
    mocker.patch.object(pooch, "os_cache", return_value=tmp_path)
    return tmp_path


def test_load_test_data(DATA_DIR):
    """Downloaded connectivity data should match the local test data."""
    expected_data = Connections.from_files(
        node_info=DATA_DIR / "small-nodes.csv",
        edge_table=DATA_DIR / "small-edge-table.csv",
        edge_info=DATA_DIR / "small-edge-meta.csv",
    )
    loaded_test_data = load_test_data()

    assert_frame_equal(loaded_test_data.nodes, expected_data.nodes)
    assert_frame_equal(loaded_test_data.edge_info, expected_data.edge_info)
    assert loaded_test_data.network.nodes() == expected_data.network.nodes()
    assert (
        loaded_test_data.network.weighted_edge_list()
        == expected_data.network.weighted_edge_list()
    )


@pytest.mark.parametrize(
    "sex, edge_info_rows",
    [
        pytest.param("f", 48146, id="female"),
        pytest.param("male", 323514, id="male"),
    ],
)
def test_load_data(sex, edge_info_rows):
    """Download the real dataset and check its graph and metadata."""
    loaded_data = load_data(sex=sex)

    assert isinstance(loaded_data, Connections)
    assert loaded_data.nodes.height == 834
    assert loaded_data.network.num_nodes() == 834
    assert loaded_data.network.num_edges() == 74098
    assert loaded_data.nodes["region_id"][0] == "GPl_1"
    assert loaded_data.edge_info.height == edge_info_rows


def test_load_data_default_sex():
    """Load the female dataset when no sex is supplied."""
    loaded_data = load_data()
    assert loaded_data.edge_info.height == 48146


@pytest.mark.parametrize("sex", ["invalid", "F", "Female", "M", "Male", ""])
def test_load_data_invalid_sex(sex):
    """Reject unsupported sex values before attempting a download."""
    with pytest.raises(ValueError, match="sex must be"):
        load_data(sex=sex)


def test_load_refresh(mock_cache):
    """Check that refresh replaces a modified cached file."""
    load_test_data()

    url = f"{gin_url()}test-data/small-nodes.csv"
    cached_file = mock_cache / pooch.utils.unique_file_name(url)
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
