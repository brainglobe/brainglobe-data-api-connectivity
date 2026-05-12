import pandas as pd
import pytest

from brainglobe_data_api_connectivity.utils.convert import lookup_node_index

morph_dict = {"one": 1, "two": 2}


def _test_morph(region_side: str) -> int:
    return morph_dict[region_side]


@pytest.mark.parametrize(
    ["info2morph", "region_id", "error", "expected_idx"],
    [
        pytest.param("one", "Area1", None, 0),
        pytest.param("two", "Area1", None, 1),
        pytest.param("one", "Area2", None, 2),
        pytest.param("two", "Area2", ValueError, None),
    ],
)
def test_lookup_node_index(info2morph, region_id, error, expected_idx):

    info = pd.DataFrame(
        {
            "info_to_morph": [1, 2, 1],
            "region_id": ["Area1", "Area1", "Area2"],
        }
    )
    region_info = {"info_to_morph": info2morph, "region_id": region_id}

    region_to_node_heading = {"info": "info_to_morph", "id": "region_id"}
    morph_value = {"info_to_morph": _test_morph}

    if error:
        with pytest.raises(error):
            idx = lookup_node_index(
                region_info, info, region_to_node_heading, morph_value
            )
    else:
        idx = lookup_node_index(
            region_info, info, region_to_node_heading, morph_value
        )
        assert idx == expected_idx
