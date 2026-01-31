"""
Optimal bot for Carnegie Cookoff (AWAP 2026).
Architecture: Analyzer → Planner → Executor, connected by TurnContext.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController

# ── Tuning Constants ──────────────────────────────────────────────
BURN_SAFETY_BUFFER = 5
BLOCKAGE_PATIENCE = 3
URGENCY_BUFFER = 10


# ── Shared State ──────────────────────────────────────────────────
@dataclass
class TurnContext:
    turn: int = 0
    money: int = 0
    my_team: Team = Team.RED
    enemy_team: Team = Team.BLUE
    bot_ids: List[int] = field(default_factory=list)
    bot_states: Dict[int, Dict] = field(default_factory=dict)

    # Static map info (set once on turn 0)
    tile_cache: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    tile_counts: Dict[str, int] = field(default_factory=dict)
    can_cook: bool = True
    can_chop: bool = True
    map_width: int = 0
    map_height: int = 0

    # Dynamic per-turn state
    orders: List[Dict] = field(default_factory=list)
    enemy_positions: List[Tuple[int, int]] = field(default_factory=list)
    enemy_on_our_map: bool = False

    # Plate tracking
    clean_plates_on_tables: int = 0
    dirty_plates_in_sinks: int = 0
    sink_washing: bool = False

    # Pan tracking: cooker (x,y) → {"has_pan": bool, "food": Optional[dict], "cook_progress": int}
    pan_locations: Dict[Tuple[int, int], Dict] = field(default_factory=dict)

    # Counter contents: (x,y) → item dict or None
    counter_contents: Dict[Tuple[int, int], Optional[Dict]] = field(default_factory=dict)

    # Planner fills these
    assigned_jobs: Dict[int, Any] = field(default_factory=dict)

    # Enemy map snapshot (during sabotage window)
    enemy_map_snapshot: Optional[Any] = None


# ── Pathfinder ────────────────────────────────────────────────────
class Pathfinder:
    """Lazy BFS pathfinding. Computes on demand, caches per source position."""

    def __init__(self):
        self._cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self._walkable: Optional[Set[Tuple[int, int]]] = None

    def init_walkable(self, controller: RobotController, team: Team) -> None:
        """Cache the set of walkable tiles (call once on turn 0 and after map switch)."""
        m = controller.get_map(team)
        self._walkable = set()
        for x in range(m.width):
            for y in range(m.height):
                if m.is_tile_walkable(x, y):
                    self._walkable.add((x, y))
        self._cache.clear()

    def _bfs(self, start: Tuple[int, int], blocked: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[int, List[Tuple[int, int]]]]:
        """BFS from start. Returns {pos: (distance, [first_step_dx_dy])}."""
        if self._walkable is None:
            return {}

        result: Dict[Tuple[int, int], Tuple[int, List[Tuple[int, int]]]] = {}
        result[start] = (0, [])
        queue = deque([start])

        while queue:
            cx, cy = queue.popleft()
            curr_dist = result[(cx, cy)][0]
            curr_path = result[(cx, cy)][1]

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) not in self._walkable:
                        continue
                    if (nx, ny) in blocked:
                        continue
                    if (nx, ny) in result:
                        continue
                    first_step = curr_path[0] if curr_path else (dx, dy)
                    result[(nx, ny)] = (curr_dist + 1, [first_step])
                    queue.append((nx, ny))

        return result

    def get_distance(self, start: Tuple[int, int], target: Tuple[int, int],
                     blocked: Optional[Set[Tuple[int, int]]] = None) -> int:
        """Get BFS distance from start to target. Returns 9999 if unreachable."""
        blocked = blocked or set()
        cache_key = (start, frozenset(blocked))
        if cache_key not in self._cache:
            bfs_result = self._bfs(start, blocked)
            self._cache[cache_key] = {pos: dist for pos, (dist, _) in bfs_result.items()}
        distances = self._cache[cache_key]
        return distances.get(target, 9999)

    def get_next_step(self, start: Tuple[int, int], target: Tuple[int, int],
                      blocked: Optional[Set[Tuple[int, int]]] = None) -> Optional[Tuple[int, int]]:
        """Get the (dx, dy) for the first step from start toward target. None if unreachable."""
        blocked = blocked or set()
        bfs_result = self._bfs(start, blocked)
        if target not in bfs_result:
            return None
        _, path = bfs_result[target]
        return path[0] if path else None

    def get_distance_to_adjacent(self, start: Tuple[int, int], target: Tuple[int, int],
                                  blocked: Optional[Set[Tuple[int, int]]] = None) -> int:
        """Distance to any tile adjacent (Chebyshev <=1) to target. For reaching interaction range."""
        blocked = blocked or set()
        best = 9999
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                adj = (target[0] + dx, target[1] + dy)
                if adj == target:
                    if self._walkable and adj in self._walkable:
                        d = self.get_distance(start, adj, blocked)
                        best = min(best, d)
                    continue
                if self._walkable and adj in self._walkable:
                    d = self.get_distance(start, adj, blocked)
                    best = min(best, d)
        return best

    def get_step_toward_adjacent(self, start: Tuple[int, int], target: Tuple[int, int],
                                  blocked: Optional[Set[Tuple[int, int]]] = None) -> Optional[Tuple[int, int]]:
        """Get next step to reach a tile adjacent to target (for interacting with non-walkable tiles)."""
        blocked = blocked or set()
        if max(abs(start[0] - target[0]), abs(start[1] - target[1])) <= 1:
            return None  # No move needed

        best_dist = 9999
        best_step = None
        bfs_result = self._bfs(start, blocked)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                adj = (target[0] + dx, target[1] + dy)
                if adj in bfs_result:
                    dist, path = bfs_result[adj]
                    if dist < best_dist and path:
                        best_dist = dist
                        best_step = path[0]
        return best_step

    def clear_cache(self) -> None:
        self._cache.clear()
