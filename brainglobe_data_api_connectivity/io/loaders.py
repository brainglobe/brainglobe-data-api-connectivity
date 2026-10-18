"""Load connectivity data from GIN."""

from enum import StrEnum
from pathlib import Path

import pooch

from ..connections import Connections


class Sex(StrEnum):
    """Map accepted sex inputs to 'f' or 'm' dataset suffixes."""

    f = "f"
    female = "f"
    m = "m"
    male = "m"


def gin_url() -> str:
    """Return the base URL for connectivity data hosted on GIN."""
    return "https://gin.swc.ucl.ac.uk/brainglobe/data-api-connectivity/raw/master/"


def _retrieve(url: str, refresh: bool = False) -> Path:
    """Retrieve a file, optionally downloading a fresh copy.


    Pooch does not provide a built-in force download option, so this private
    helper adds that functionality by removing the cached file before
    retrieval when requested.


    Args:
        url: URL of the file to retrieve.
        refresh: If True, download a fresh copy instead of using the
            cached file. Defaults to False.

    Returns:
        Path to the downloaded or cached file.
    """
    cache_path = Path(pooch.os_cache("pooch"))
    filename = pooch.utils.unique_file_name(url)

    if refresh:
        (cache_path / filename).unlink(missing_ok=True)

    return Path(
        pooch.retrieve(
            url=url,
            path=cache_path,
            fname=filename,
        )
    )


def load_test_data(refresh: bool = False) -> Connections:
    """Load the test connectivity dataset from GIN.

    Args:
        refresh: If True, download fresh copies instead of using cached files.
            Defaults to False.
    """
    path_node_info = _retrieve(
        url=f"{gin_url()}test-data/small-nodes.csv",
        refresh=refresh,
    )

    path_edge_table = _retrieve(
        url=f"{gin_url()}test-data/small-edge-table.csv",
        refresh=refresh,
    )

    path_edge_info = _retrieve(
        url=f"{gin_url()}test-data/small-edge-meta.csv",
        refresh=refresh,
    )

    return Connections.from_files(
        node_info=path_node_info,
        edge_table=path_edge_table,
        edge_info=path_edge_info,
    )


def load_data(sex: str = "f", refresh: bool = False) -> Connections:
    """Load data from GIN for the requested sex.

    Loads a Connections object from GIN, constructed from CSV files
    derived from the supplementary Excel datasets of Swanson et al. (2025).
    The CSV representation provides a more manageable format for working
    with the original large Excel datasets.

    Reference:
        Swanson, L. W., Hahn, J. D., & Sporns, O. (2025).
        The intrinsic neuronal network of the central nervous system
        and its modular (subsystem) architecture in a mammal.
        Proceedings of the National Academy of Sciences, 122(40),
        e2519768122. https://doi.org/10.1073/pnas.2519768122

    Args:
        sex: Sex of the dataset. Accepts "f" or "female", and "m" or
            "male". Defaults to "f".
        refresh: Whether to download fresh copies instead of using
            cached files. Defaults to False.


    Raises:
        ValueError: If sex is not one of the supported values.
    """
    try:
        sex_suffix = Sex[sex].value
    except KeyError:
        raise ValueError(
            'sex must be "f" or "female", or "m" or "male".'
        ) from None

    dataset = f"CNS2{sex_suffix}"
    dataset_url = f"{gin_url()}data/{dataset}/{dataset}"

    path_node_info = _retrieve(
        url=f"{dataset_url}_node_info.csv",
        refresh=refresh,
    )

    path_edge_table = _retrieve(
        url=f"{dataset_url}_edge_table.csv",
        refresh=refresh,
    )

    path_edge_info = _retrieve(
        url=f"{dataset_url}_edge_info.csv",
        refresh=refresh,
    )

    return Connections.from_files(
        node_info=path_node_info,
        edge_table=path_edge_table,
        edge_info=path_edge_info,
        edge_info_from_col="origin_region_idx",
        edge_info_to_col="termination_region_idx",
    )
