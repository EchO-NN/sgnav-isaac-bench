from isaac_bench.dataset.category_normalizer import normalize_category


def test_instance_suffix_stripping():
    assert normalize_category("chair_001") == "chair"
    assert normalize_category("chair.002") == "chair"
    assert normalize_category("sofa-copy3") == "sofa"
    assert normalize_category("dining-table_12") == "dining_table"


def test_aliases():
    assert normalize_category("couch") == "sofa"
    assert normalize_category("closestool_0000") == "toilet"
    assert normalize_category("basin_0001") == "sink"

