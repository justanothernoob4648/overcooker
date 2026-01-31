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
