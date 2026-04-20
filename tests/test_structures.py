from glyphgen.structures import component_boxes, layout_heatmap
from glyphgen.types import StructureType


def test_component_boxes_return_two_regions() -> None:
    boxes = component_boxes(StructureType.LEFT_RIGHT, image_size=128)
    assert len(boxes) == 2
    assert boxes[0].x0 < boxes[0].x1
    assert boxes[1].x0 < boxes[1].x1


def test_layout_heatmap_shape() -> None:
    heatmap = layout_heatmap(StructureType.TOP_BOTTOM, image_size=64)
    assert heatmap.shape == (2, 64, 64)
    assert float(heatmap.max()) == 1.0
