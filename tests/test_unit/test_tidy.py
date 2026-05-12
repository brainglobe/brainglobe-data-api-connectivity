from pathlib import Path

import pandas as pd
import pytest

from brainglobe_data_api_connectivity.utils.tidy import (
    consolidate_duplicates,
    rename_columns,
)


@pytest.mark.parametrize(
    ["name", "expected"],
    [
        pytest.param("blank space", "blank_space", id="space"),
        pytest.param("new\nline", "new_line", id="newline"),
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
        pytest.param("5___mix.£d tHiNg$___", "mix_d_thing_5", id="mixed"),
    ],
)
def test_rename_columns(name, expected):
    """Test whether rename_columns works as expected."""
    renamed = rename_columns(pd.Index([name]))[0]
    assert renamed == expected


@pytest.mark.parametrize(
    ["files", "error"],
    [
        pytest.param(
            {
                "a_info.csv": pd.DataFrame({"x": [1, 2]}),
            },
            None,
            id="single file",
        ),
        pytest.param(
            {
                "a_info.csv": pd.DataFrame({"x": [1, 2]}),
                "b_info.csv": pd.DataFrame({"x": [1, 2]}),
            },
            None,
            id="duplicates",
        ),
        pytest.param(
            {
                "a_info.csv": pd.DataFrame({"x": [1, 2]}),
                "b_info.csv": pd.DataFrame({"x": [9, 9]}),
            },
            ValueError,
            id="mismatched files",
        ),
        pytest.param(
            {
                "a_info.csv": pd.DataFrame({"x": [1, 2]}),
                "b_info.csv": pd.DataFrame({"x": [1, 2]}),
                "node_info.csv": pd.DataFrame({"x": [1, 2]}),
            },
            None,
            id="duplicates + node_info file",
        ),
    ],
)
def test_consolidate_duplicates(tmp_path: Path, files, error):
    """Test consolidation of duplicates

    consolidate_duplicates should
    - keep one file with the right name and content
    - raise an error for mismatched files."""

    for name, df in files.items():
        df.to_csv(tmp_path / name, index=False)

    pattern = "*info.csv"
    target_name = "node_info.csv"

    if error is not None:
        with pytest.raises(error):
            consolidate_duplicates(pattern, tmp_path, target_name)
        return

    else:
        consolidate_duplicates(pattern, tmp_path, target_name)

        remaining = sorted(p.name for p in tmp_path.glob(pattern))
        assert len(remaining) == 1
        assert remaining == [target_name]

        content = pd.read_csv(tmp_path / target_name)
        expected = pd.DataFrame({"x": [1, 2]})

        assert content.equals(expected)
