"""Load connectivity data from GIN."""

import pooch

from ..connections import Connections

SEX_SUFFIXES = {
    "f": "f",
    "female": "f",
    "m": "m",
    "male": "m",
}


def gin_url() -> str:
    """Return the base URL for connectivity data hosted on GIN."""
    return "https://gin.swc.ucl.ac.uk/brainglobe/data-api-connectivity/raw/master/"


def load_test_data() -> Connections:
    """Load the test connectivity dataset from GIN."""
    path_node_info = pooch.retrieve(
        url=f"{gin_url()}test-data/small-nodes.csv",
    )

    path_edge_table = pooch.retrieve(
        url=f"{gin_url()}test-data/small-edge-table.csv",
    )

    path_edge_info = pooch.retrieve(
        url=f"{gin_url()}test-data/small-edge-meta.csv",
    )

    return Connections.from_files(
        node_info=path_node_info,
        edge_table=path_edge_table,
        edge_info=path_edge_info,
    )


def load_data(sex: str = "F") -> Connections:
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
        sex: "F", "female", "M", or "male" (case-insensitive).
            Defaults to "F".

    Raises:
        ValueError: If sex is not one of the supported values.
    """
    sex = sex.lower()

    if sex not in SEX_SUFFIXES:
        raise ValueError('sex must be "F", "female", "M", or "male".')

    dataset = f"CNS2{SEX_SUFFIXES[sex]}"
    dataset_url = f"{gin_url()}data/{dataset}/{dataset}"

    path_node_info = pooch.retrieve(
        url=f"{dataset_url}_node_info.csv",
    )

    path_edge_table = pooch.retrieve(
        url=f"{dataset_url}_edge_table.csv",
    )

    path_edge_info = pooch.retrieve(
        url=f"{dataset_url}_edge_info.csv",
    )

    return Connections.from_files(
        node_info=path_node_info,
        edge_table=path_edge_table,
        edge_info=path_edge_info,
        edge_info_from_col="origin_region_idx",
        edge_info_to_col="termination_region_idx",
    )
