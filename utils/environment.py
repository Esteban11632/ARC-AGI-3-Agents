ACTION_TO_DIRECTION = {
    1: (-1, 0),   # up
    2: (1, 0),    # down
    3: (0, -1),   # left
    4: (0, 1),    # right
}

def get_adjacent_colors(cells, grid, dr, dc) -> frozenset[int]:
        """Colors in the cells immediately outside the object's edge in direction (dr, dc)."""
        cell_set = set(cells)
        colors = set()
        h, w = grid.shape
        for r, c in cells:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_set:
                continue              # interior cell, not the edge facing that direction
            if 0 <= nr < h and 0 <= nc < w:
                colors.add(int(grid[nr, nc]))
            else:
                colors.add(-1)        # -1 = out of bounds / border wall
        return frozenset(colors)