"""Tidy excel input data"""

from pathlib import Path

import pandas as pd


def rename_columns(columns: pd.Index) -> pd.Index:
    """Rename column names.

    Changes column names so they never start with a diget and can only contain
     - lowercase characters a-z
     - digits 0-9
     - underscores
    """
    cleaned = (
        columns.str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.replace(
            r"^(\d+)([a-z0-9_]+)$", r"\2_\1", regex=True
        )  # move leading digits to the end
        .str.replace(r"_+", "_", regex=True)  # collapse multiple _
        .str.strip("_")  # trim any underscores from start or end
    )
    return pd.Index(cleaned)


def consolidate_duplicates(
    pattern: str,
    folder: str | Path,
    output_name: str,
) -> Path | None:
    """Consolidate duplicate CSV files in a folder.

    Finds all files matching a given pattern, verifies they are duplicates,
    and replaces them with a single file saved as `output_name`.

    Parameters
    pattern: str
        Pattern used find duplicates (e.g. '*.csv').
    folder: str | Path
        Directory in which to search for matching files.
    output_name: str
        Name of the consolidated output file to create.

    Returns
    target: Path | None
        Path to the consolidated file, or None if no files matched.
    """
    folder = Path(folder)

    target = folder / output_name
    if target.exists():
        target.unlink()

    files = sorted(folder.glob(pattern))
    if not files:
        return None

    # verify content of all files match
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if not pd.read_csv(files[i]).equals(pd.read_csv(files[j])):
                raise ValueError(
                    f"Files {files[i].name} and {files[j].name} do not match."
                )

    # consolidate
    files[0].rename(target)
    for f in files[1:]:
        f.unlink()

    return target
