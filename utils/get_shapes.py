"""Shape predicates for grid objects defined by cell coordinates."""

from collections.abc import Sequence

def bbox_dimensions(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int]:
    """Return (width, height, area) for a cell list."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    width = max(cols) - min(cols) + 1
    height = max(rows) - min(rows) + 1
    return width, height, len(cells)


def is_single_cell(cells: Sequence[tuple[int, int]]) -> bool:
    return len(cells) == 1


def is_line(cells: Sequence[tuple[int, int]]) -> bool:
    """True if all cells lie on one straight line (horizontal, vertical, or diagonal)."""
    if len(cells) < 2:
        return False

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}

    if len(rows) == 1 or len(cols) == 1:
        return True

    if len({r - c for r, c in cells}) == 1:
        return True

    if len({r + c for r, c in cells}) == 1:
        return True

    return False


def is_rectangle(cells: Sequence[tuple[int, int]]) -> bool:
    """True if cells completely fill their bounding box."""
    width, height, area = bbox_dimensions(cells)
    return area == width * height


def is_square(cells: Sequence[tuple[int, int]]) -> bool:
    width, height, _ = bbox_dimensions(cells)
    return is_rectangle(cells) and width == height


def is_l_shape(cells: Sequence[tuple[int, int]]) -> bool:
    """True for a 3-cell bent tromino (L)."""
    _, _, area = bbox_dimensions(cells)
    return area == 3 and not is_line(cells)


def _neighbor_counts(cells: Sequence[tuple[int, int]]) -> dict[tuple[int, int], int]:
    cell_set = set(cells)
    counts: dict[tuple[int, int], int] = {}

    for r, c in cells:
        count = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in cell_set:
                count += 1
        counts[(r, c)] = count

    return counts


def is_cross(cells: Sequence[tuple[int, int]]) -> bool:
    """True for a plus/cross shape with one center and four orthogonal arms."""
    _, _, area = bbox_dimensions(cells)
    if area != 5:
        return False

    counts = _neighbor_counts(cells)
    return list(counts.values()).count(4) == 1


def is_t_shape(cells: Sequence[tuple[int, int]]) -> bool:
    """True for a T tetromino (4 cells)."""
    _, _, area = bbox_dimensions(cells)
    if area != 4 or is_line(cells):
        return False

    counts = _neighbor_counts(cells)
    return list(counts.values()).count(3) == 1
