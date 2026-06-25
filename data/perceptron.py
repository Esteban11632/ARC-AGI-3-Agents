from dataclasses import dataclass
from collections import deque
import numpy as np
from enum import Enum

from utils.get_shapes import (
    is_cross, 
    is_l_shape, 
    is_line, 
    is_rectangle, 
    is_square, 
    is_t_shape
)

class ShapeType(Enum):
    UNKNOWN = 0
    LINE = 1
    SQUARE = 2
    RECTANGLE = 3
    L_SHAPE = 4
    CROSS = 5
    T_SHAPE = 6

@dataclass
class GridObject:
    object_id: int
    color: int
    previous_color: int
    cells: list[tuple[int, int]]
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    width: int
    height: int
    area: int
    fill_ratio: float
    area_ratio: float
    is_large_region: bool
    shape_type: ShapeType = ShapeType.UNKNOWN
    delta_x: int = 0
    delta_y: int = 0
    color_changed: bool = False
    rotation_delta: float = 0
    rotation_changed: bool = False
    position_changed: bool = False
    matched: int = -1


class ObjectExtractor:
    def __init__(self, grid_height = 64, grid_width = 64):
        self.frames = []
        self.grids = []
        self.grid_height = grid_height
        self.grid_width = grid_width

    def get_neighbors_8(self, r, c):
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        for dr, dc in directions:
            nr = r + dr
            nc = c + dc

            if 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                yield nr, nc

    def flood_fill(self, grid, sr, sc, visited):
        color = int(grid[sr, sc])

        # Cells to visit
        q = deque()
        q.append((sr, sc))
        visited[sr, sc] = True

        # Cells visited
        cells = []

        while q:
            r, c = q.popleft()
            cells.append((r, c))

            for nr, nc in self.get_neighbors_8(r, c):
                if visited[nr, nc]:
                    continue

                if int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    q.append((nr, nc))

        return cells

    def extract(self, grid):
        grid = np.array(grid)
        self.grid_height, self.grid_width = grid.shape

        visited = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        objects = []
        object_id = 0

        for r in range(self.grid_height):
            for c in range(self.grid_width):

                if visited[r, c]:
                    continue

                color = int(grid[r, c])

                cells = self.flood_fill(
                    grid=grid,
                    sr=r,
                    sc=c,
                    visited=visited
                )
                
                rows = [r for r, _ in cells]
                cols = [c for _, c in cells]

                min_row = min(rows)
                min_col = min(cols)
                max_row = max(rows)
                max_col = max(cols)

                center = self.get_center(rows, cols, cells)

                area = len(cells)

                width = max(cols) - min(cols) + 1
                height = max(rows) - min(rows) + 1

                bbox_area = width * height
                fill_ratio = area / bbox_area

                shape_type = self.get_shape_type(cells)
                area_ratio = area / (self.grid_height * self.grid_width)
                is_large_region = area_ratio > 0.35
                obj = GridObject(
                        object_id=object_id,
                        color=color,
                        previous_color=color,
                        cells=cells,
                        center=center,
                        bbox=(min_row, min_col, max_row, max_col),
                        width=width,
                        height=height,
                        area=area,
                        fill_ratio=fill_ratio,
                        shape_type=shape_type,
                        area_ratio=area_ratio,
                        is_large_region=is_large_region,
                    )

                objects.append(obj)
                object_id += 1

        self.frames.append(objects)
        self.grids.append(grid.copy())

        return objects
    
    def get_center(self, rows, cols, cells):

        center_row = sum(rows) / len(cells)
        center_col = sum(cols) / len(cells)

        return center_row, center_col
    
    def normalize_cells(self, cells):
        # Shifts the cells to the top-left corner so position on the grid doesn't matter
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)

        # Use a set to automatically remove duplicate coordinates if they exist
        normalized = {(r - min_r, c - min_c) for r, c in cells}
        return tuple(sorted(normalized))

    def rotate_90(self, cells):
        return [(c, -r) for r, c in cells]

    def abstract_shape_signature(self, obj):
        shape_type = obj.shape_type

        if shape_type == ShapeType.LINE:
            return (ShapeType.LINE, obj.area)

        if shape_type == ShapeType.SQUARE:
            return (ShapeType.SQUARE, obj.area)

        if shape_type == ShapeType.RECTANGLE:
            return (ShapeType.RECTANGLE, min(obj.width, obj.height), max(obj.width, obj.height))

        if shape_type == ShapeType.L_SHAPE:
            return (ShapeType.L_SHAPE, obj.area)

        if shape_type == ShapeType.CROSS:
            return (ShapeType.CROSS, obj.area)

        return (ShapeType.UNKNOWN, obj.area)

    def get_shape_type(self, cells):
        if is_line(cells):
            return ShapeType.LINE

        if is_square(cells):
            return ShapeType.SQUARE

        if is_rectangle(cells):
            return ShapeType.RECTANGLE

        if is_l_shape(cells):
            return ShapeType.L_SHAPE

        if is_cross(cells):
            return ShapeType.CROSS

        if is_t_shape(cells):
            return ShapeType.T_SHAPE

        return ShapeType.UNKNOWN

    def track_objects(self, max_distance=3):
        if len(self.frames) < 2:
            return
    
        current = self.frames[-1]
        previous = self.frames[-2]
        pairs = self.match_objects(
            current_frame=current,
            previous_frame=previous,
            max_distance=max_distance,
            previous_grid=self.grids[-2],
            current_grid=self.grids[-1],
        )

        for current_obj, previous_obj in pairs:
            current_obj.matched = previous_obj.object_id
            self.get_movement(current_obj, previous_obj)
            self.get_previous_color(current_obj, previous_obj)
            current_obj.rotation_delta = self.rotation_delta(previous_obj.cells, current_obj.cells)
            current_obj.rotation_changed = current_obj.rotation_delta != 0
        
    def _center_distance(self, a, b) -> float:
        return abs(a.center[0] - b.center[0]) + abs(a.center[1] - b.center[1])

    def _cells_overlap(self, a, b) -> bool:
        return bool(set(a.cells) & set(b.cells))

    def match_objects(
        self,
        current_frame,
        previous_frame,
        max_distance,
    ):
        """Pair objects using shape + motion cues (colors may change in-game).
        
        Color match is used as a tiebreaker: same-color pairs are preferred over
        color-changed pairs at equal distance, but color never hard-blocks a match.
        """
        candidates = []
        for ci, current_obj in enumerate(current_frame):
            for pj, previous_obj in enumerate(previous_frame):
                if self.abstract_shape_signature(current_obj) != self.abstract_shape_signature(previous_obj):
                    continue

                dist = self._center_distance(current_obj, previous_obj)
                if dist > max_distance:
                    continue

                color_bonus = 0.5 if current_obj.color == previous_obj.color else 0
                # A different-colored object at the exact same position is almost
                # certainly a new/revealed tile, not the original object changing color.
                # Penalize heavily so a moved-and-recolored object beats it.
                same_pos_diff_color = dist == 0 and current_obj.color != previous_obj.color
                score = dist - color_bonus + (max_distance + 1 if same_pos_diff_color else 0)
                candidates.append((score, ci, pj))

        matches = []
        used_current = set()
        used_previous = set()

        for score, ci, pj in sorted(candidates):
            if ci in used_current or pj in used_previous:
                continue
            matches.append((current_frame[ci], previous_frame[pj]))
            used_current.add(ci)
            used_previous.add(pj)

        return matches
    
    def get_previous_color(self, current_object, previous_object):
        current_object.previous_color = previous_object.color
        current_object.color_changed = current_object.previous_color != current_object.color
    
    def get_movement(self, current_object, past_object):
        current_object.delta_x = current_object.center[0] - past_object.center[0]
        current_object.delta_y = current_object.center[1] - past_object.center[1]
        current_object.position_changed = (
            current_object.delta_x != 0 or current_object.delta_y != 0
        )
    
    def rotation_delta(self, previous_cells, current_cells) -> int:
        """Degrees clockwise from previous orientation to current."""
        previous = list(self.normalize_cells(previous_cells))
        target = self.normalize_cells(current_cells)

        for steps in range(4):
            if self.normalize_cells(previous) == target:
                return steps * 90
            previous = self.rotate_90(previous)

        return 0

"""grid_before = [
    [0, 2, 2, 2],
    [0, 2, 2, 2],
    [0, 0, 0, 3],
    [3, 0, 0, 3],
]

grid_after = [
    [0, 2, 2, 0],
    [0, 2, 2, 0],
    [0, 2, 2, 3],
    [3, 0, 0, 3],
]

extractor = ObjectExtractor()
extractor.extract(grid_before)
extractor.extract(grid_after)
extractor.track_objects()

previous_objects = extractor.frames[-2]
print(previous_objects[1])

current_objects = extractor.frames[-1]
print(current_objects[1])"""