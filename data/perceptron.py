from dataclasses import dataclass
from collections import deque
import numpy as np
import numpy as np
import torch

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
    is_single_cell: bool
    is_line: bool
    is_rectangle: bool
    is_square: bool
    shape_signature: tuple[tuple[int,int]]
    area_ratio: float
    is_large_region: bool
    delta_x: int = 0
    delta_y: int = 0
    color_changed: bool = False
    rotation_delta: float = 0
    rotation_changed: bool = False
    position_changed: bool = False
    matched: bool = False


class ObjectExtractor:
    def __init__(self, grid_height = 64, grid_width = 64):
        self.frames = []
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

                is_single_cell = area == 1
                is_line = width == 1 or height == 1
                is_rectangle = area == bbox_area
                is_square = is_rectangle and width == height
                shape_sig = self.shape_signature(cells)
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
                        is_single_cell=is_single_cell,
                        is_line=is_line,
                        is_rectangle=is_rectangle,
                        is_square=is_square,
                        shape_signature=shape_sig,
                        area_ratio=area_ratio,
                        is_large_region=is_large_region,
                    )

                objects.append(obj)
                object_id += 1

        self.frames.append(objects)

        return objects
    
    def get_center(self, rows, cols, cells):

        center_row = sum(rows) / len(cells)
        center_col = sum(cols) / len(cells)

        return center_row, center_col
    
    def normalize_cells(self, cells):
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)

        normalized = [(r - min_r, c - min_c) for r, c in cells]
        return tuple(sorted(normalized))


    def rotate_90(self, cells):
        # rotate around origin: (r, c) -> (c, -r)
        return [(c, -r) for r, c in cells]


    def shape_signature(self, cells):
        # Step 1: normalize original cells
        cells = list(self.normalize_cells(cells))

        versions = []

        current = cells
        for _ in range(4):
            normalized = self.normalize_cells(current)
            versions.append(normalized)
            current = self.rotate_90(current)

        # Pick canonical version
        return min(versions)

    def track_objects(self, max_distance=3):
        if len(self.frames) < 2:
            return
        
        current = self.frames[-1]
        previous = self.frames[-2]
        pairs = self.match_objects(current_frame=current, previous_frame=previous, max_distance=max_distance)

        for current_obj, previous_obj in pairs:
            current_obj.matched = True
            self.get_movement(current_obj, previous_obj)
            self.get_previous_color(current_obj, previous_obj)
            current_obj.rotation_delta = self.rotation_delta(previous_obj.cells, current_obj.cells)
            current_obj.rotation_changed = current_obj.rotation_delta != 0
        
    def match_objects(self, current_frame, previous_frame, max_distance):

        used_previous = set()
        matches = []

        for current_obj in current_frame:
            best_prev = None
            best_dist = float("inf")

            for j, previous_obj in enumerate(previous_frame):
                if j in used_previous:
                    continue

                if current_obj.shape_signature != previous_obj.shape_signature:
                    continue

                delta_x = current_obj.center[0] - previous_obj.center[0]
                delta_y = current_obj.center[1] - previous_obj.center[1]

                dist = abs(delta_x) + abs(delta_y)

                if dist < best_dist and dist <= max_distance:
                    best_dist = dist
                    best_prev = j

            if best_prev is not None:
                previous_obj = previous_frame[best_prev]

                matches.append((current_obj, previous_obj))

                used_previous.add(best_prev)

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

    def objects_to_model_input(self, objects):

        features = []

        for obj in objects:
            min_row, min_col, max_row, max_col = obj.bbox

            features.append([
                            obj.color,

                            # Normalized center (0 to 1)
                            obj.center[0] / self.grid_height,
                            obj.center[1] / self.grid_width,

                            # Normalized bounding box (0 to 1)
                            min_row / self.grid_height,
                            min_col / self.grid_width,
                            max_row / self.grid_height,
                            max_col / self.grid_width,

                            # Normalized width and height (0 to 1)
                            obj.width / self.grid_width,
                            obj.height / self.grid_height,

                            # Normalized area (0 to 1)
                            obj.area / (self.grid_height * self.grid_width),

                            # Fraction of bbox occupied by object
                            obj.fill_ratio,

                            float(obj.is_single_cell),
                            float(obj.is_line),
                            float(obj.is_rectangle),
                            float(obj.is_square),
                            float(obj.area_ratio),
                            float(obj.is_large_region),
                            obj.delta_x / self.grid_height,
                            obj.delta_y / self.grid_width,
                            float(obj.color_changed),
                            obj.rotation_delta / 360.0,
                            float(obj.rotation_changed),
                            float(obj.position_changed),
                        ])

        features = torch.tensor(features, dtype=torch.float32)

        return features

grid_before = [
    [0, 2, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 3],
    [3, 0, 0, 3],
]

grid_after = [
    [0, 0, 2, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 3],
    [3, 0, 0, 3],
]

extractor = ObjectExtractor()
extractor.extract(grid_before)
extractor.extract(grid_after)
extractor.track_objects()

objects = extractor.frames[-1]
features = extractor.objects_to_model_input(objects)
print(features)
for obj in objects:
    print(obj)
    print("--------------------------------")