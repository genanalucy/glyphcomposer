from glyphgen.data.decomposition import load_decomposition_csv


def test_load_decomposition_csv() -> None:
    records = load_decomposition_csv("assets/decompositions_sample.csv")
    assert records
    assert records[0].target_char
    assert records[0].component_a
