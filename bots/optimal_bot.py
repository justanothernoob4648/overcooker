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


# ── Analyzer ──────────────────────────────────────────────────────
class Analyzer:
    """Scans the map and populates TurnContext."""

    def __init__(self):
        self._initialized = False
        self._ctx = TurnContext()

    def analyze(self, controller: RobotController, pathfinder: Pathfinder) -> TurnContext:
        """Build TurnContext for the current turn."""
        ctx = self._ctx
        ctx.turn = controller.get_turn()
        ctx.my_team = controller.get_team()
        ctx.enemy_team = controller.get_enemy_team()
        ctx.money = controller.get_team_money(ctx.my_team)
        ctx.bot_ids = controller.get_team_bot_ids(ctx.my_team)

        ctx.bot_states = {}
        for bid in ctx.bot_ids:
            state = controller.get_bot_state(bid)
            if state:
                ctx.bot_states[bid] = state

        if not self._initialized:
            self._scan_static_map(controller, pathfinder, ctx)
            self._initialized = True

        self._scan_dynamic_state(controller, ctx)
        self._track_enemies(controller, ctx)
        self._scout_enemy_map(controller, ctx)

        return ctx

    def _scan_static_map(self, controller: RobotController, pathfinder: Pathfinder, ctx: TurnContext) -> None:
        """One-time full map scan on turn 0."""
        m = controller.get_map(ctx.my_team)
        ctx.map_width = m.width
        ctx.map_height = m.height

        tile_names = ["SHOP", "COOKER", "COUNTER", "SINK", "SINKTABLE", "SUBMIT", "TRASH", "BOX"]
        ctx.tile_cache = {name: [] for name in tile_names}
        ctx.tile_counts = {name: 0 for name in tile_names}

        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                name = tile.tile_name
                if name in ctx.tile_cache:
                    ctx.tile_cache[name].append((x, y))
                    ctx.tile_counts[name] += 1

        ctx.can_cook = ctx.tile_counts.get("COOKER", 0) > 0
        ctx.can_chop = ctx.tile_counts.get("COUNTER", 0) > 0

        pathfinder.init_walkable(controller, ctx.my_team)

    def _scan_dynamic_state(self, controller: RobotController, ctx: TurnContext) -> None:
        """Update dynamic state each turn."""
        ctx.orders = controller.get_orders(ctx.my_team)

        # Scan cookers for pan/food state
        ctx.pan_locations = {}
        for (x, y) in ctx.tile_cache.get("COOKER", []):
            tile = controller.get_tile(ctx.my_team, x, y)
            if tile is None:
                continue
            item = getattr(tile, "item", None)
            cook_progress = getattr(tile, "cook_progress", 0)
            if item is not None and hasattr(item, "food"):
                food_dict = None
                if item.food is not None:
                    food_dict = {
                        "food_name": getattr(item.food, "food_name", None),
                        "cooked_stage": getattr(item.food, "cooked_stage", 0),
                    }
                ctx.pan_locations[(x, y)] = {
                    "has_pan": True,
                    "food": food_dict,
                    "cook_progress": cook_progress,
                    "is_empty": item.food is None,
                }
            else:
                ctx.pan_locations[(x, y)] = {
                    "has_pan": False,
                    "food": None,
                    "cook_progress": 0,
                    "is_empty": True,
                }

        # Scan counters for contents
        ctx.counter_contents = {}
        for (x, y) in ctx.tile_cache.get("COUNTER", []):
            tile = controller.get_tile(ctx.my_team, x, y)
            if tile is None:
                continue
            item = getattr(tile, "item", None)
            if item is not None:
                ctx.counter_contents[(x, y)] = controller.item_to_public_dict(item)
            else:
                ctx.counter_contents[(x, y)] = None

        # Scan sinks and sink tables
        ctx.clean_plates_on_tables = 0
        ctx.dirty_plates_in_sinks = 0
        ctx.sink_washing = False
        for (x, y) in ctx.tile_cache.get("SINKTABLE", []):
            tile = controller.get_tile(ctx.my_team, x, y)
            if tile is not None:
                ctx.clean_plates_on_tables += getattr(tile, "num_clean_plates", 0)
        for (x, y) in ctx.tile_cache.get("SINK", []):
            tile = controller.get_tile(ctx.my_team, x, y)
            if tile is not None:
                ctx.dirty_plates_in_sinks += getattr(tile, "num_dirty_plates", 0)
                if getattr(tile, "using", False):
                    ctx.sink_washing = True

    def _track_enemies(self, controller: RobotController, ctx: TurnContext) -> None:
        """Track enemy bot positions."""
        enemy_ids = controller.get_team_bot_ids(ctx.enemy_team)
        ctx.enemy_positions = []
        ctx.enemy_on_our_map = False
        for eid in enemy_ids:
            state = controller.get_bot_state(eid)
            if state and state.get("map_team") == ctx.my_team.name:
                ctx.enemy_on_our_map = True
                ctx.enemy_positions.append((state["x"], state["y"]))

    def _scout_enemy_map(self, controller: RobotController, ctx: TurnContext) -> None:
        """During sabotage window, snapshot the enemy map."""
        switch_info = controller.get_switch_info()
        ctx.enemy_map_snapshot = None

        if not switch_info.get("window_active", False):
            return
        if switch_info.get("my_team_switched", False):
            return

        enemy_map = controller.get_map(ctx.enemy_team)
        snapshot = {"cookers": {}, "counters": {}}

        for x in range(enemy_map.width):
            for y in range(enemy_map.height):
                tile = enemy_map.tiles[x][y]
                name = tile.tile_name
                if name == "COOKER":
                    item = getattr(tile, "item", None)
                    cook_progress = getattr(tile, "cook_progress", 0)
                    food_info = None
                    if item and hasattr(item, "food") and item.food:
                        food_info = {
                            "food_name": item.food.food_name,
                            "cooked_stage": item.food.cooked_stage,
                            "buy_cost": getattr(item.food, "buy_cost", 0),
                        }
                    snapshot["cookers"][(x, y)] = {
                        "has_pan": item is not None and hasattr(item, "food"),
                        "food": food_info,
                        "cook_progress": cook_progress,
                    }
                elif name == "COUNTER":
                    item = getattr(tile, "item", None)
                    if item is not None:
                        item_info = {"type": type(item).__name__}
                        if hasattr(item, "food_name"):
                            item_info["food_name"] = item.food_name
                            item_info["buy_cost"] = getattr(item, "buy_cost", 0)
                        snapshot["counters"][(x, y)] = item_info

        ctx.enemy_map_snapshot = snapshot


# ── Dependency Graph ──────────────────────────────────────────────
class NodeStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class GraphNode:
    """A single task in the dependency graph."""

    def __init__(self, node_id: str, action: str, **kwargs):
        self.node_id = node_id
        self.action = action
        self.status = NodeStatus.PENDING
        self.dependencies: List[str] = []
        self.params = kwargs
        self.priority: int = 0
        self.deadline: Optional[int] = None
        self.result_location: Optional[Tuple[int, int]] = None

    def is_ready(self, completed_nodes: Set[str]) -> bool:
        return all(dep in completed_nodes for dep in self.dependencies)


class DependencyGraph:
    """Dependency graph for fulfilling a single order."""

    def __init__(self, order_id: int):
        self.order_id = order_id
        self.nodes: Dict[str, GraphNode] = {}
        self.completed: Set[str] = set()

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def mark_done(self, node_id: str, result_location: Optional[Tuple[int, int]] = None) -> None:
        self.nodes[node_id].status = NodeStatus.DONE
        self.completed.add(node_id)
        if result_location:
            self.nodes[node_id].result_location = result_location

    def mark_failed(self, node_id: str) -> None:
        self.nodes[node_id].status = NodeStatus.FAILED

    def get_ready_nodes(self) -> List[GraphNode]:
        ready = []
        for node in self.nodes.values():
            if node.status in (NodeStatus.PENDING, NodeStatus.IN_PROGRESS):
                if node.is_ready(self.completed):
                    ready.append(node)
        return ready

    def is_complete(self) -> bool:
        return all(n.status == NodeStatus.DONE for n in self.nodes.values())

    def has_failed(self) -> bool:
        return any(n.status == NodeStatus.FAILED for n in self.nodes.values())


def build_order_graph(order: Dict, ctx: TurnContext) -> DependencyGraph:
    """Build a dependency graph for fulfilling the given order."""
    graph = DependencyGraph(order["order_id"])
    required_foods = order["required"]

    FOOD_INFO = {
        "EGG":     {"needs_chop": False, "needs_cook": True,  "cost": 20},
        "ONIONS":  {"needs_chop": True,  "needs_cook": False, "cost": 30},
        "MEAT":    {"needs_chop": True,  "needs_cook": True,  "cost": 80},
        "NOODLES": {"needs_chop": False, "needs_cook": False, "cost": 40},
        "SAUCE":   {"needs_chop": False, "needs_cook": False, "cost": 10},
    }

    FOOD_ENUM = {
        "EGG": FoodType.EGG,
        "ONIONS": FoodType.ONIONS,
        "MEAT": FoodType.MEAT,
        "NOODLES": FoodType.NOODLES,
        "SAUCE": FoodType.SAUCE,
    }

    all_food_final_nodes: List[str] = []

    for i, food_name in enumerate(required_foods):
        info = FOOD_INFO.get(food_name, {"needs_chop": False, "needs_cook": False, "cost": 0})
        prefix = f"food_{i}_{food_name}"

        buy_node = GraphNode(f"{prefix}_buy", "buy", item=FOOD_ENUM.get(food_name), food_name=food_name)
        graph.add_node(buy_node)
        prev_node_id = buy_node.node_id

        if info["needs_chop"] and ctx.can_chop:
            place_for_chop = GraphNode(f"{prefix}_place_chop", "place", target_tile="COUNTER")
            place_for_chop.dependencies = [prev_node_id]
            graph.add_node(place_for_chop)

            chop_node = GraphNode(f"{prefix}_chop", "chop", target_tile="COUNTER")
            chop_node.dependencies = [place_for_chop.node_id]
            graph.add_node(chop_node)
            prev_node_id = chop_node.node_id

        if info["needs_cook"] and ctx.can_cook:
            if info["needs_chop"]:
                pickup_for_cook = GraphNode(f"{prefix}_pickup_for_cook", "pickup", target_tile="COUNTER")
                pickup_for_cook.dependencies = [prev_node_id]
                graph.add_node(pickup_for_cook)
                prev_node_id = pickup_for_cook.node_id

            cook_node = GraphNode(f"{prefix}_start_cook", "start_cook", target_tile="COOKER")
            cook_node.dependencies = [prev_node_id]
            graph.add_node(cook_node)

            take_node = GraphNode(f"{prefix}_take_pan", "take_from_pan", target_tile="COOKER")
            take_node.dependencies = [cook_node.node_id]
            take_node.priority = 10
            graph.add_node(take_node)

            place_cooked = GraphNode(f"{prefix}_place_cooked", "place", target_tile="COUNTER")
            place_cooked.dependencies = [take_node.node_id]
            graph.add_node(place_cooked)
            prev_node_id = place_cooked.node_id

        elif not info["needs_chop"]:
            place_raw = GraphNode(f"{prefix}_place_raw", "place", target_tile="COUNTER")
            place_raw.dependencies = [prev_node_id]
            graph.add_node(place_raw)
            prev_node_id = place_raw.node_id

        all_food_final_nodes.append(prev_node_id)

    plate_node = GraphNode("get_plate", "get_plate")
    graph.add_node(plate_node)

    assembly_deps = all_food_final_nodes + [plate_node.node_id]
    assemble_node = GraphNode("assemble", "assemble", num_foods=len(required_foods))
    assemble_node.dependencies = assembly_deps
    graph.add_node(assemble_node)

    submit_node = GraphNode("submit", "submit", target_tile="SUBMIT")
    submit_node.dependencies = [assemble_node.node_id]
    graph.add_node(submit_node)

    return graph