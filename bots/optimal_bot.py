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


# ── Planner ───────────────────────────────────────────────────────
class Planner:
    """Scores orders and assigns jobs to bots."""

    def __init__(self):
        self.active_graphs: Dict[int, DependencyGraph] = {}
        self._reserved_money: int = 0
        self._reserved_orders: Set[int] = set()

    def plan(self, ctx: TurnContext, pathfinder: Pathfinder) -> Dict[int, DependencyGraph]:
        """Assign or re-assign jobs. Returns bot_id -> DependencyGraph."""
        for bid in ctx.bot_ids:
            graph = self.active_graphs.get(bid)
            if graph is not None and not graph.is_complete() and not graph.has_failed():
                continue

            best_order, best_score = self._pick_best_order(ctx, pathfinder, bid)

            if best_order is not None:
                new_graph = build_order_graph(best_order, ctx)
                self.active_graphs[bid] = new_graph
                self._reserved_orders.add(best_order["order_id"])
                self._reserved_money += self._order_cost(best_order)
            else:
                self.active_graphs.pop(bid, None)

        return self.active_graphs

    def force_replan(self, ctx: TurnContext, pathfinder: Pathfinder) -> None:
        self._reserved_orders.clear()
        self._reserved_money = 0
        for bid in list(self.active_graphs.keys()):
            graph = self.active_graphs.get(bid)
            if graph is not None and graph.has_failed():
                self.active_graphs.pop(bid, None)
        self.plan(ctx, pathfinder)

    def _pick_best_order(self, ctx: TurnContext, pathfinder: Pathfinder,
                          bot_id: int) -> Tuple[Optional[Dict], float]:
        best_order = None
        best_score = -1.0
        available_money = ctx.money - self._reserved_money

        for order in ctx.orders:
            if not order.get("is_active", False):
                continue
            if order.get("completed_turn") is not None:
                continue
            if order["order_id"] in self._reserved_orders:
                continue

            score = self._score_order(order, ctx, pathfinder, bot_id, available_money)
            if score > best_score:
                best_score = score
                best_order = order

        return best_order, best_score

    def _score_order(self, order: Dict, ctx: TurnContext, pathfinder: Pathfinder,
                      bot_id: int, available_money: int) -> float:
        cost = self._order_cost(order)
        estimated_turns = self._estimate_turns(order, ctx, pathfinder, bot_id)

        if estimated_turns <= 0:
            return -1.0

        if cost > available_money:
            income_wait = (cost - available_money)
            estimated_turns += income_wait

        feasibility = (order["expires_turn"] - ctx.turn) - estimated_turns
        if feasibility < 0:
            return -1.0

        urgency = max(0.0, 1.0 - (feasibility / URGENCY_BUFFER))
        effective_value = order["reward"] + (order["penalty"] * urgency) - cost
        if effective_value <= 0:
            return 0.0

        return effective_value / estimated_turns

    def _order_cost(self, order: Dict) -> int:
        COSTS = {"EGG": 20, "ONIONS": 30, "MEAT": 80, "NOODLES": 40, "SAUCE": 10}
        cost = sum(COSTS.get(f, 0) for f in order["required"])
        cost += ShopCosts.PLATE.buy_cost
        needs_cook = any(f in ("EGG", "MEAT") for f in order["required"])
        if needs_cook:
            cost += ShopCosts.PAN.buy_cost
        return cost

    def _estimate_turns(self, order: Dict, ctx: TurnContext, pathfinder: Pathfinder,
                         bot_id: int) -> int:
        bot_state = ctx.bot_states.get(bot_id)
        if not bot_state:
            return 9999

        bot_pos = (bot_state["x"], bot_state["y"])
        total = 0

        shops = ctx.tile_cache.get("SHOP", [])
        counters = ctx.tile_cache.get("COUNTER", [])
        cookers = ctx.tile_cache.get("COOKER", [])
        submits = ctx.tile_cache.get("SUBMIT", [])

        if not shops or not submits:
            return 9999

        shop = shops[0]
        submit = submits[0]

        FOOD_INFO = {
            "EGG":     {"needs_chop": False, "needs_cook": True},
            "ONIONS":  {"needs_chop": True,  "needs_cook": False},
            "MEAT":    {"needs_chop": True,  "needs_cook": True},
            "NOODLES": {"needs_chop": False, "needs_cook": False},
            "SAUCE":   {"needs_chop": False, "needs_cook": False},
        }

        dist_to_shop = pathfinder.get_distance_to_adjacent(bot_pos, shop)
        total += dist_to_shop

        cook_count = 0
        for food_name in order["required"]:
            info = FOOD_INFO.get(food_name, {})
            if counters:
                total += pathfinder.get_distance_to_adjacent(shop, counters[0]) + 1
            total += 1

            if info.get("needs_chop"):
                total += 2

            if info.get("needs_cook"):
                cook_count += 1
                if cookers:
                    total += pathfinder.get_distance_to_adjacent(counters[0] if counters else shop, cookers[0])
                total += 1

        num_cookers = max(1, ctx.tile_counts.get("COOKER", 1))
        if cook_count > 0:
            cook_batches = -(-cook_count // num_cookers)
            total += max(0, cook_batches * GameConstants.COOK_PROGRESS - total // 2)

        if ctx.clean_plates_on_tables <= 0:
            if ctx.dirty_plates_in_sinks > 0:
                total += GameConstants.PLATE_WASH_PROGRESS + 3
            else:
                total += 2

        total += len(order["required"]) + 1

        if counters:
            total += pathfinder.get_distance_to_adjacent(counters[0], submit)
        total += 1

        return max(1, total)

    def evaluate_sabotage(self, ctx: TurnContext) -> float:
        if ctx.enemy_map_snapshot is None:
            return -1.0

        total_value = 0.0

        for loc, info in ctx.enemy_map_snapshot.get("cookers", {}).items():
            food = info.get("food")
            if food:
                base_cost = food.get("buy_cost", 0)
                cook_progress = info.get("cook_progress", 0)
                prep_value = base_cost + cook_progress * 2
                total_value += prep_value

        for loc, info in ctx.enemy_map_snapshot.get("counters", {}).items():
            buy_cost = info.get("buy_cost", 0)
            total_value += buy_cost

        if total_value <= 0:
            return -1.0

        est_turns = 20
        return total_value / est_turns


# ── Executor ──────────────────────────────────────────────────────
class Executor:
    """Translates dependency graph nodes into per-turn move + action calls."""

    def __init__(self):
        self.ingredient_locations: Dict[str, Tuple[int, int]] = {}
        self.cook_locations: Dict[str, Tuple[int, int]] = {}
        self.cook_start_turns: Dict[str, int] = {}
        self.plate_location: Optional[Tuple[int, int]] = None
        self.wait_counters: Dict[str, int] = {}

    def execute_turn(self, bot_id: int, graph: DependencyGraph,
                      ctx: TurnContext, pathfinder: Pathfinder,
                      controller: RobotController) -> bool:
        if graph is None or graph.is_complete():
            return False

        bot_state = ctx.bot_states.get(bot_id)
        if not bot_state:
            return False
        bot_pos = (bot_state["x"], bot_state["y"])
        holding = bot_state.get("holding")

        blocked = set()
        for bid, bs in ctx.bot_states.items():
            if bid != bot_id:
                blocked.add((bs["x"], bs["y"]))
        for pos in ctx.enemy_positions:
            blocked.add(pos)

        burn_node = self._check_burn_urgent(graph, ctx)
        if burn_node and holding is not None:
            return self._drop_held_item(bot_id, bot_pos, ctx, pathfinder, controller, blocked)

        ready = graph.get_ready_nodes()
        if not ready:
            return False

        if burn_node and burn_node in ready:
            node = burn_node
        else:
            ready.sort(key=lambda n: -n.priority)
            node = ready[0]

        node.status = NodeStatus.IN_PROGRESS
        return self._execute_node(bot_id, node, graph, bot_pos, holding,
                                   ctx, pathfinder, controller, blocked)

    def _check_burn_urgent(self, graph: DependencyGraph, ctx: TurnContext) -> Optional[GraphNode]:
        for node in graph.nodes.values():
            if node.action != "take_from_pan":
                continue
            if node.status in (NodeStatus.DONE, NodeStatus.FAILED):
                continue
            cook_key = node.node_id.replace("_take_pan", "_start_cook")
            cooker_pos = self.cook_locations.get(cook_key)
            if cooker_pos is None:
                continue
            pan_info = ctx.pan_locations.get(cooker_pos)
            if pan_info and pan_info.get("food"):
                progress = pan_info.get("cook_progress", 0)
                if progress >= GameConstants.COOK_PROGRESS and progress < GameConstants.BURN_PROGRESS - BURN_SAFETY_BUFFER:
                    continue
                if progress >= GameConstants.BURN_PROGRESS - BURN_SAFETY_BUFFER:
                    node.priority = 100
                    return node
        return None

    def _execute_node(self, bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked):
        if node.action == "buy":
            return self._do_buy(bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked)
        elif node.action == "place":
            return self._do_place(bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked)
        elif node.action == "chop":
            return self._do_chop(bot_id, node, graph, bot_pos, ctx, pathfinder, controller, blocked)
        elif node.action == "pickup":
            return self._do_pickup(bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked)
        elif node.action == "start_cook":
            return self._do_start_cook(bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked)
        elif node.action == "take_from_pan":
            return self._do_take_from_pan(bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked)
        elif node.action == "get_plate":
            return self._do_get_plate(bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked)
        elif node.action == "assemble":
            return self._do_assemble(bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked)
        elif node.action == "submit":
            return self._do_submit(bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked)
        return False

    def _move_adjacent(self, bot_id, bot_pos, target, pathfinder, controller, blocked):
        dist = max(abs(bot_pos[0] - target[0]), abs(bot_pos[1] - target[1]))
        if dist <= 1:
            return True
        step = pathfinder.get_step_toward_adjacent(bot_pos, target, blocked)
        if step:
            controller.move(bot_id, step[0], step[1])
        return False

    def _find_empty_counter(self, ctx):
        for pos, item in ctx.counter_contents.items():
            if item is None:
                return pos
        return None

    def _find_cooker_with_empty_pan(self, ctx):
        for pos, info in ctx.pan_locations.items():
            if info.get("has_pan") and info.get("is_empty"):
                return pos
        return None

    def _drop_held_item(self, bot_id, bot_pos, ctx, pathfinder, controller, blocked):
        counter = self._find_empty_counter(ctx)
        if counter is None:
            return False
        if self._move_adjacent(bot_id, bot_pos, counter, pathfinder, controller, blocked):
            return controller.place(bot_id, counter[0], counter[1])
        return True

    def _do_buy(self, bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked):
        if holding is not None:
            return self._drop_held_item(bot_id, bot_pos, ctx, pathfinder, controller, blocked)
        shops = ctx.tile_cache.get("SHOP", [])
        if not shops:
            graph.mark_failed(node.node_id)
            return False
        target = shops[0]
        if self._move_adjacent(bot_id, bot_pos, target, pathfinder, controller, blocked):
            item = node.params.get("item")
            if item and controller.buy(bot_id, item, target[0], target[1]):
                graph.mark_done(node.node_id)
                return True
        return True

    def _do_place(self, bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked):
        if holding is None:
            graph.mark_done(node.node_id)
            return False
        target_tile = node.params.get("target_tile", "COUNTER")
        if target_tile == "COUNTER":
            target = self._find_empty_counter(ctx)
        elif target_tile == "COOKER":
            target = self._find_cooker_with_empty_pan(ctx)
        else:
            target = None
        if target is None:
            graph.mark_failed(node.node_id)
            return False
        if self._move_adjacent(bot_id, bot_pos, target, pathfinder, controller, blocked):
            if controller.place(bot_id, target[0], target[1]):
                graph.mark_done(node.node_id, result_location=target)
                self.ingredient_locations[node.node_id] = target
                return True
        return True

    def _do_chop(self, bot_id, node, graph, bot_pos, ctx, pathfinder, controller, blocked):
        place_node_id = node.dependencies[0] if node.dependencies else None
        target = self.ingredient_locations.get(place_node_id)
        if target is None:
            for pos, item in ctx.counter_contents.items():
                if item and item.get("type") == "Food" and not item.get("chopped", False):
                    target = pos
                    break
        if target is None:
            graph.mark_failed(node.node_id)
            return False
        if self._move_adjacent(bot_id, bot_pos, target, pathfinder, controller, blocked):
            if controller.chop(bot_id, target[0], target[1]):
                graph.mark_done(node.node_id, result_location=target)
                self.ingredient_locations[node.node_id] = target
                return True
        return True

    def _do_pickup(self, bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked):
        if holding is not None:
            return self._drop_held_item(bot_id, bot_pos, ctx, pathfinder, controller, blocked)
        dep_id = node.dependencies[0] if node.dependencies else None
        target = self.ingredient_locations.get(dep_id)
        if target is None:
            graph.mark_failed(node.node_id)
            return False
        if self._move_adjacent(bot_id, bot_pos, target, pathfinder, controller, blocked):
            if controller.pickup(bot_id, target[0], target[1]):
                graph.mark_done(node.node_id)
                return True
        return True

    def _do_start_cook(self, bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked):
        if holding is None:
            return False
        cooker = self._find_cooker_with_empty_pan(ctx)
        if cooker is None:
            graph.mark_failed(node.node_id)
            return False
        if self._move_adjacent(bot_id, bot_pos, cooker, pathfinder, controller, blocked):
            if controller.start_cook(bot_id, cooker[0], cooker[1]):
                graph.mark_done(node.node_id, result_location=cooker)
                self.cook_locations[node.node_id] = cooker
                self.cook_start_turns[node.node_id] = ctx.turn
                return True
            if controller.place(bot_id, cooker[0], cooker[1]):
                graph.mark_done(node.node_id, result_location=cooker)
                self.cook_locations[node.node_id] = cooker
                self.cook_start_turns[node.node_id] = ctx.turn
                return True
        return True

    def _do_take_from_pan(self, bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked):
        if holding is not None:
            return self._drop_held_item(bot_id, bot_pos, ctx, pathfinder, controller, blocked)
        cook_key = None
        for dep in node.dependencies:
            if dep in self.cook_locations:
                cook_key = dep
                break
        cooker = self.cook_locations.get(cook_key) if cook_key else None
        if cooker is None:
            for pos, info in ctx.pan_locations.items():
                if info.get("food") and info["food"].get("cooked_stage", 0) >= 1:
                    cooker = pos
                    break
        if cooker is None:
            pan_info = None
            if cook_key and cook_key in self.cook_locations:
                pan_info = ctx.pan_locations.get(self.cook_locations[cook_key])
            if pan_info and pan_info.get("food"):
                stage = pan_info["food"].get("cooked_stage", 0)
                if stage < 1:
                    return False
            return False
        if self._move_adjacent(bot_id, bot_pos, cooker, pathfinder, controller, blocked):
            if controller.take_from_pan(bot_id, cooker[0], cooker[1]):
                graph.mark_done(node.node_id)
                return True
        return True

    def _do_get_plate(self, bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked):
        if holding is not None:
            return self._drop_held_item(bot_id, bot_pos, ctx, pathfinder, controller, blocked)
        if ctx.clean_plates_on_tables > 0:
            tables = ctx.tile_cache.get("SINKTABLE", [])
            if tables:
                target = tables[0]
                if self._move_adjacent(bot_id, bot_pos, target, pathfinder, controller, blocked):
                    if controller.take_clean_plate(bot_id, target[0], target[1]):
                        graph.mark_done(node.node_id)
                        return True
                return True
        if ctx.dirty_plates_in_sinks > 0:
            sinks = ctx.tile_cache.get("SINK", [])
            if sinks:
                target = sinks[0]
                if self._move_adjacent(bot_id, bot_pos, target, pathfinder, controller, blocked):
                    controller.wash_sink(bot_id, target[0], target[1])
                    return True
                return True
        shops = ctx.tile_cache.get("SHOP", [])
        if shops:
            target = shops[0]
            if self._move_adjacent(bot_id, bot_pos, target, pathfinder, controller, blocked):
                if controller.buy(bot_id, ShopCosts.PLATE, target[0], target[1]):
                    graph.mark_done(node.node_id)
                    return True
            return True
        return False

    def _do_assemble(self, bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked):
        if holding is None:
            if self.plate_location:
                target = self.plate_location
                if self._move_adjacent(bot_id, bot_pos, target, pathfinder, controller, blocked):
                    if controller.pickup(bot_id, target[0], target[1]):
                        self.plate_location = None
                return True
            return False
        if holding and holding.get("type") == "Plate":
            for node_id, loc in self.ingredient_locations.items():
                item = ctx.counter_contents.get(loc)
                if item and item.get("type") == "Food":
                    if self._move_adjacent(bot_id, bot_pos, loc, pathfinder, controller, blocked):
                        if controller.add_food_to_plate(bot_id, loc[0], loc[1]):
                            ctx.counter_contents[loc] = None
                    return True
            graph.mark_done(node.node_id)
            return True
        if holding and holding.get("type") == "Food":
            for pos, item in ctx.counter_contents.items():
                if item and item.get("type") == "Plate" and not item.get("dirty", True):
                    if self._move_adjacent(bot_id, bot_pos, pos, pathfinder, controller, blocked):
                        if controller.add_food_to_plate(bot_id, pos[0], pos[1]):
                            pass
                    return True
        return False

    def _do_submit(self, bot_id, node, graph, bot_pos, holding, ctx, pathfinder, controller, blocked):
        if holding is None or holding.get("type") != "Plate":
            return False
        submits = ctx.tile_cache.get("SUBMIT", [])
        if not submits:
            graph.mark_failed(node.node_id)
            return False
        target = submits[0]
        if self._move_adjacent(bot_id, bot_pos, target, pathfinder, controller, blocked):
            if controller.submit(bot_id, target[0], target[1]):
                graph.mark_done(node.node_id)
                return True
        return True


# ── BotPlayer (Entry Point) ──────────────────────────────────────
class BotPlayer:
    """Main entry point. Created once per game, play_turn called each turn."""

    def __init__(self, map_copy):
        self.map = map_copy
        self.pathfinder = Pathfinder()
        self.analyzer = Analyzer()
        self.planner = Planner()
        self.executors: Dict[int, Executor] = {}

    def play_turn(self, controller: RobotController):
        try:
            self._play_turn_inner(controller)
        except Exception as e:
            pass

    def _play_turn_inner(self, controller: RobotController):
        ctx = self.analyzer.analyze(controller, self.pathfinder)
        graphs = self.planner.plan(ctx, self.pathfinder)

        for bot_id in ctx.bot_ids:
            graph = graphs.get(bot_id)
            if graph is None:
                self._do_idle(bot_id, ctx, controller)
                continue

            if bot_id not in self.executors:
                self.executors[bot_id] = Executor()

            executor = self.executors[bot_id]

            if not hasattr(executor, '_current_order') or executor._current_order != graph.order_id:
                self.executors[bot_id] = Executor()
                executor = self.executors[bot_id]
                executor._current_order = graph.order_id

            executor.execute_turn(bot_id, graph, ctx, self.pathfinder, controller)

    def _do_idle(self, bot_id: int, ctx: TurnContext, controller: RobotController):
        bot_state = ctx.bot_states.get(bot_id)
        if not bot_state:
            return

        bot_pos = (bot_state["x"], bot_state["y"])
        holding = bot_state.get("holding")

        if holding:
            counter = None
            for pos, item in ctx.counter_contents.items():
                if item is None:
                    counter = pos
                    break
            if counter:
                dist = max(abs(bot_pos[0] - counter[0]), abs(bot_pos[1] - counter[1]))
                if dist <= 1:
                    controller.place(bot_id, counter[0], counter[1])
                else:
                    step = self.pathfinder.get_step_toward_adjacent(bot_pos, counter)
                    if step:
                        controller.move(bot_id, step[0], step[1])
            return

        if ctx.dirty_plates_in_sinks > 0 and ctx.clean_plates_on_tables == 0:
            sinks = ctx.tile_cache.get("SINK", [])
            if sinks:
                target = sinks[0]
                dist = max(abs(bot_pos[0] - target[0]), abs(bot_pos[1] - target[1]))
                if dist <= 1:
                    controller.wash_sink(bot_id, target[0], target[1])
                else:
                    step = self.pathfinder.get_step_toward_adjacent(bot_pos, target)
                    if step:
                        controller.move(bot_id, step[0], step[1])
                return

        shops = ctx.tile_cache.get("SHOP", [])
        if shops:
            target = shops[0]
            dist = max(abs(bot_pos[0] - target[0]), abs(bot_pos[1] - target[1]))
            if dist > 1:
                step = self.pathfinder.get_step_toward_adjacent(bot_pos, target)
                if step:
                    controller.move(bot_id, step[0], step[1])