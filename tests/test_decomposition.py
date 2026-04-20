from glyphgen.data.decomposition import DecompositionRecord
from glyphgen.types import StructureType


def test_decomposition_record_from_row() -> None:
    record = DecompositionRecord.from_row(
        {
            "target_char": "狗",
            "structure": "left_right",
            "component_a": "犭",
            "component_b": "句",
        }
    )
    assert record.target_char == "狗"
    assert record.structure is StructureType.LEFT_RIGHT
