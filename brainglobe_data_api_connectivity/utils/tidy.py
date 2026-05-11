from pathlib import Path

import pandas as pd


def rename_columns(columns: pd.Index) -> pd.Index:
    """Rename column names.

    Column names can only contain
     - lowercase characters a-z
     - digits 0-9
     - underscores

    Names cannot start with a digit.
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
    """Remove duplicate files.

    - find files matching a pattern
    - ensure they are identical
    - keep one under a new name.
    """
    folder = Path(folder)

    files = sorted(folder.glob(pattern))
    if not files:
        return None

    target = folder / output_name
    if target.exists():
        target.unlink()

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
