from typing import List, Tuple

def boxes_overlap(b1, b2, expand=0):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    x1 -= expand
    y1 -= expand
    w1 += 2 * expand
    h1 += 2 * expand

    return not (
        x1 + w1 < x2 or
        x2 + w2 < x1 or
        y1 + h1 < y2 or
        y2 + h2 < y1
    )

def merge_boxes(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y

    return (x, y, w, h)

def consolidate_boxes(
    boxes: List[Tuple[int, int, int, int]],
    expand: int = 20
) -> List[Tuple[int, int, int, int]]:
    merged = []

    for box in boxes:
        has_merged = False
        for i, m in enumerate(merged):
            if boxes_overlap(box, m, expand=expand):
                merged[i] = merge_boxes(box, m)
                has_merged = True
                break

        if not has_merged:
            merged.append(box)

    return merged