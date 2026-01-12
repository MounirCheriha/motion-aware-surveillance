from src.motion.roi_utils import consolidate_boxes


def test_consolidate_merges_overlapping_boxes():
    b1 = (10, 10, 20, 20)
    b2 = (25, 15, 20, 20)  # overlaps with b1 when expand used

    merged = consolidate_boxes([b1, b2], expand=5)

    assert len(merged) == 1
    x, y, w, h = merged[0]
    assert x <= 10 and y <= 10
    assert w >= 35 and h >= 25


def test_consolidate_keeps_separate_non_overlapping():
    b1 = (0, 0, 10, 10)
    b2 = (50, 50, 10, 10)

    merged = consolidate_boxes([b1, b2], expand=5)

    assert len(merged) == 2
