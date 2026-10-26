"""Grid map + A* path planning + path-to-actions for AI2-THOR."""
import heapq
import math

GRID = 0.25  # meters per cell (AI2-THOR default step size)
# direction (dx, dz) -> robot heading in degrees
HEADINGS = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}


def to_cell(x, z):
    return (round(x / GRID), round(z / GRID))


def to_xz(cell):
    return (cell[0] * GRID, cell[1] * GRID)


class GridMap:
    """Walkable cells come from AI2-THOR reachable positions; everything else is an obstacle."""

    def __init__(self, reachable):
        self.free = {to_cell(p["x"], p["z"]) for p in reachable}

    def neighbors(self, cell):
        for dx, dz in HEADINGS:
            n = (cell[0] + dx, cell[1] + dz)
            if n in self.free:
                yield n

    def goal_cells(self, obj_xz, reach=1.0):
        """Free cells close enough for the robot to reach the object."""
        return {c for c in self.free if math.dist(to_xz(c), obj_xz) <= reach}


def astar(grid, start, goals):
    """Shortest 4-connected path from start to any goal cell (list of cells, or None)."""
    if not goals:
        return None
    h = lambda c: min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goals)
    open_set = [(h(start), 0, start)]
    came_from = {start: None}
    cost = {start: 0}
    while open_set:
        _, g, cur = heapq.heappop(open_set)
        if cur in goals:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1]
        if g > cost[cur]:
            continue
        for n in grid.neighbors(cur):
            if g + 1 < cost.get(n, float("inf")):
                cost[n] = g + 1
                came_from[n] = cur
                heapq.heappush(open_set, (g + 1 + h(n), g + 1, n))
    return None


def path_to_actions(path, heading):
    """Turn a list of cells into RotateLeft / RotateRight / MoveAhead actions."""
    actions = []
    for a, b in zip(path, path[1:]):
        target = HEADINGS[(b[0] - a[0], b[1] - a[1])]
        turn = (target - heading) % 360
        if turn == 90:
            actions.append("RotateRight")
        elif turn == 270:
            actions.append("RotateLeft")
        elif turn == 180:
            actions += ["RotateRight", "RotateRight"]
        heading = target
        actions.append("MoveAhead")
    return actions, heading
