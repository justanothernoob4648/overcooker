# robot_controller.py
"""
RobotController API for the competitive cooking game.

Rules enforced here:
- Per turn: each bot may move at most once AND perform at most one action.
- Movement is single-step and must be within Chebyshev distance 1 (|dx|<=1, |dy|<=1).
- All actions target a tile within Chebyshev distance 1 of the bot (including own tile).
- Money is TEAM-SHARED (GameState.team_money[Team]).
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from game_constants import Team, FoodType, ShopCosts, GameConstants
from map import Map
from tiles import Tile, Counter, Sink, SinkTable, Cooker, Trash, Submit, Shop, Box
from item import Item, Food, Plate, Pan

from game_state import GameState

from typing import Union

Buyable = Union[FoodType, ShopCosts]



class RobotController:
    '''Class where robots can call the specified PUBLIC actions to alter game state'''

    def __init__(self, team: Team, game_state: GameState):
        self.__team = team
        self.__game_state = game_state

        self.__last_seen_turn: int = game_state.turn #curr turn
        self.__moves_left: Dict[int, int] = {}
        self.__actions_left: Dict[int, int] = {}
        self.__refresh_turn_budgets()

    # ----------------------------
    # Turn helpers
    # ----------------------------
    def __refresh_turn_budgets(self) -> None:
        '''can only move once AND act once per turn'''
        for bot_id in self.get_team_bot_ids(self.__team):
            self.__moves_left[bot_id] = 1
            self.__actions_left[bot_id] = 1

    def __ensure_turn(self) -> None:
        '''refresh with checks for turn state ie if new turn, add new movements'''
        if self.__game_state.turn != self.__last_seen_turn:
            self.__last_seen_turn = self.__game_state.turn
            self.__refresh_turn_budgets()

    def __consume_move(self, bot_id: int) -> bool:
        '''make a movement action'''
        self.__ensure_turn() #refresh
        
        if self.__moves_left.get(bot_id, 0) <= 0:
            #self.__warn(f"bot {bot_id} has already moved this turn")
            return False
        
        self.__moves_left[bot_id] -= 1
        return True

    def __consume_action(self, bot_id: int) -> bool:
        '''acts'''
        self.__ensure_turn() #refresh

        if self.__actions_left.get(bot_id, 0) <= 0:
            #self.__warn(f"bot {bot_id} has already acted this turn")
            return False
        
        self.__actions_left[bot_id] -= 1
        return True

    # ----------------------------
    # General safe state access
    # ----------------------------

    def get_turn(self) -> int:
        return self.__game_state.turn

    def get_team(self) -> Team:
        return self.__team

    def get_enemy_team(self) -> Team:
        return Team.RED if self.__team == Team.BLUE else Team.BLUE

    def get_map(self, team: Team) -> Map:
        '''Deep copy for the user'''
        return copy.deepcopy(self.__game_state.get_map(team))

    def get_orders(self, team: Team) -> List[Dict[str, Any]]:
        '''returns list of dictionaries (each order is represented by the dictionary)'''
        res = []
        for o in self.__game_state.orders.get(team, []):
            res.append(
                {
                    "order_id": o.order_id,
                    "required": [ft.food_name for ft in o.required],
                    "created_turn": o.created_turn,
                    "expires_turn": o.expires_turn,
                    "reward": o.reward,
                    "penalty": o.penalty,
                    "claimed_by": o.claimed_by,
                    "completed_turn": o.completed_turn,
                    "is_active": o.is_active(self.__game_state.turn),
                }
            )
        return res

    def get_team_bot_ids(self, team: Team) -> List[int]:
        '''returns bot ids of a specified team as a list'''
        return [bot_id for bot_id, b in self.__game_state.bots.items() if b.team == team]

    def get_team_money(self, team: Team) -> int:
        '''returns money for a team (yours and your opponent's)'''
        return self.__game_state.get_team_money(team)

    def get_bot_state(self, bot_id: int) -> Optional[Dict[str, Any]]:
        '''returns a dictionary of bot state as a dictionary; note holding provides a dictionary too'''
        try:
            b = self.__game_state.get_bot(bot_id)
        except Exception:
            #self.__warn(f"Invalid bot_id {bot_id}")
            return None

        if b is None:
            return None
        
        return {
            "bot_id": b.bot_id,
            "team": b.team.name,
            "x": b.x,
            "y": b.y,
            "team_money": self.__game_state.get_team_money(b.team),
            "holding": self.item_to_public_dict(b.holding),
            "map_team": getattr(b, "map_team", b.team).name,
        }

    def get_tile(self, team: Team, x: int, y: int) -> Optional[Tile]:
        '''Get the tile at a specific x, y'''
        try:
            t = self.__game_state.get_tile(team, x, y)
            return copy.deepcopy(t)
        
        except Exception:
            return None

    # ----------------------------
    # targeting helpers
    # ----------------------------

    @staticmethod
    def __chebyshev_dist(x0: int, y0: int, x1: int, y1: int) -> int:
        '''chess king distance'''
        return max(abs(x0 - x1), abs(y0 - y1))

    def __resolve_target_tile(self, bot_id: int, label: str, target_x: Optional[int], target_y: Optional[int]) -> Optional[Tuple[int, int, Tile]]:
        '''checks if target is good'''

        b = self.__safe_get_bot(bot_id)
        if b is None:
            return None

        target_x = b.x if target_x is None else target_x
        target_y = b.y if target_y is None else target_y

        if self.__chebyshev_dist(b.x, b.y, target_x, target_y) > 1:
            #self.__warn(f"{label} failed: target ({target_x},{target_y}) too far from bot {bot_id} at ({b.x},{b.y})")
            return None

        m = self.__game_state.get_map(b.map_team)
        if not m.in_bounds(target_x, target_y):
            #self.__warn(f"{label} failed : target ({target_x},{target_y}) is out of bounds")
            return None

        tile = self.__game_state.get_tile(b.map_team, target_x, target_y)
        return (target_x, target_y, tile)

    # ----------------------------
    # Movement helpers
    # ----------------------------

    def can_move(self, bot_id: int, dx: int, dy: int) -> bool:
        '''can move or not bot by (dx, dy) or not'''
        b = self.__safe_get_bot(bot_id)

        if b is None:
            return False
        if max(abs(dx), abs(dy)) > 1 or (dx == 0 and dy == 0):
            return False
        
        #returns the private internal checker after main checks for modularity
        return self.__can_move_internal(b.map_team, b.x, b.y, dx, dy)


    def move(self, bot_id: int, dx: int, dy: int) -> bool:
        '''actually moves, True if move succeeds; False otherwise'''
        b = self.__safe_get_bot(bot_id)

        if b is None:
            return False
        
        if not self.__consume_move(bot_id):
            return False
        
        if max(abs(dx), abs(dy)) > 1 or (dx == 0 and dy == 0):
            #self.__warn(f"move() failed: bot {bot_id} illegal step ({dx},{dy}); must be chebyshev distance 1")
            return False
        
        if not self.__can_move_internal(b.map_team, b.x, b.y, dx, dy):
            #self.__warn(f"move() failed: illegal move bot {bot_id} from ({b.x},{b.y}) by ({dx},{dy})")
            return False
        
        #move the bot through game state
        if not self.__game_state.move_bot(bot_id, dx, dy):
            #self.__warn(f"move() failed: occupied/blocked with movement of bot {bot_id} to ({b.x+dx},{b.y+dy})")
            pass

        return True


    # ----------------------------
    # botwise inventory interactions
    # ----------------------------

    def pickup(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''bot picks up from target x, target y location; box pickup special'''

        b = self.__safe_get_bot(bot_id)

        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False
        if b.holding is not None:
            #self.__warn(f"pickup() failed: bot {bot_id} already holding something")
            return False

        #check validity
        tgt = self.__resolve_target_tile(bot_id, "pickup()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        #CONSIDER BOX
        if isinstance(tile, Box):
            if getattr(tile, "count", 0) <= 0 or getattr(tile, "item", None) is None:
                # enforce invariant
                tile.count = 0
                tile.item = None
                #self.__warn(f"pickup() failed: BOX at ({target_x},{target_y}) is empty for bot {bot_id}")
                return False

            #give bot a new deepcopy of the stored prototype
            b.holding = copy.deepcopy(tile.item)
            tile.count -= 1
            if tile.count <= 0:
                tile.count = 0
                tile.item = None
            return True

        item = getattr(tile, "item", None)
        if item is None:
            #self.__warn(f"pickup() failed: nothing to pick up at ({target_x},{target_y}) for bot {bot_id}")
            return False

        b.holding = item
        tile.item = None

        return True

    def place(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''bot places to target x, target y location; box place and food on pan in cooker is special'''
        b = self.__safe_get_bot(bot_id)

        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False
        if b.holding is None:
            #self.__warn(f"place() failed: bot {bot_id} holding nothing")
            return False

        tgt = self.__resolve_target_tile(bot_id, "place()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt
        

        #COOKER SPECIAL CASE for pan with food on cooker
        #swap current cooker pan into bot's holding space
        #if placed pan has cookable food, then we init the cook ticks
        #pan swap OR food in plate
        if isinstance(tile, Cooker):
            # if bot is holding a pan, then we swap pans
            if isinstance(b.holding, Pan):
                held_pan: Pan = b.holding
                old_pan = tile.item if isinstance(getattr(tile, "item", None), Pan) else None 

                # DON'T ALLOW SWAP if it is currently cooking right now
                if isinstance(old_pan, Pan) and old_pan.food is not None:
                    #self.__warn(f"place() failed: cooker at ({target_x},{target_y}) is busy; old pan has food")
                    return False

                #else, just swap
                tile.item = held_pan
                b.holding = old_pan 

                #if the placed pan has food, then we start the cook
                if isinstance(tile.item, Pan) and isinstance(tile.item.food, Food) and tile.item.food.can_cook:
                    self.__set_cook_progress_for_food(tile, tile.item.food)
                else:
                    tile.cook_progress = 0

                return True

            #bot holds food and places the food into the pan
            if isinstance(b.holding, Food):
                pan = tile.item
                #is there pan?
                if not isinstance(pan, Pan):
                    #self.__warn(f"place() failed: cooker at ({target_x},{target_y}) missing pan for food")
                    return False
                
                #is pan empty
                if pan.food is not None:
                    #self.__warn(f"place() failed: pan at ({target_x},{target_y}) is already occupied")
                    return False
                
                #is food valid for cooking?
                if not b.holding.can_cook:
                    #self.__warn(f"place() failed: food {b.holding.food_name} cannot be cooked")
                    return False

                #move food from hand to pan
                pan.food = b.holding
                b.holding = None

                #init cook progress based on teh food
                self.__set_cook_progress_for_food(tile, pan.food)
                return True

            #not the cases above, so fail
            #self.__warn(f"place() failed: must hold Pan or cookable Food for cooker at ({target_x},{target_y})")
            return False

        #BOX SPECIAL CASE HERE WHERE WE PLACE THE BOX
        if isinstance(tile, Box):
            #enforce the invariant
            tile.enforce_invar()

            #empty box means we accept anything
            if tile.count == 0:
                tile.item = b.holding
                tile.count = 1
                b.holding = None
                return True

            #non-empty means only accept same kind
            if tile.item is None:
                tile.item = b.holding
                tile.count = 1
                b.holding = None
                return True

            if self.__item_signature(tile.item) != self.__item_signature(b.holding):
                #self.__warn(f"place() failed: box tile at ({target_x},{target_y}) stores a different item type")
                return False

            tile.count += 1
            b.holding = None
            return True

        if not hasattr(tile, "item"):
            #self.__warn(f"place() failed: tile at ({target_x},{target_y}) cannot hold items for bot {bot_id}")
            return False
        if getattr(tile, "item") is not None:
            #self.__warn(f"place() failed: tile at ({target_x},{target_y}) already has an item for bot {bot_id}")
            return False

        tile.item = b.holding
        b.holding = None
        return True

    def trash(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        b = self.__safe_get_bot(bot_id)

        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False
        if b.holding is None:
            #self.__warn(f"trash() failed: bot {bot_id} holding onto nothing")
            return False

        tgt = self.__resolve_target_tile(bot_id, "trash()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        if not isinstance(tile, Trash):
            #self.__warn(f"trash() failed: target ({target_x},{target_y}) is not trash tile for bot {bot_id}")
            return False

        if isinstance(b.holding, Plate):
            b.holding = Plate([], False) #clean plate
        elif isinstance(b.holding, Pan):
            b.holding = Pan(None) #empty pan
        else:
            b.holding = None
        return True

    # ----------------------------
    # Shop / economy (TEAM money)
    # ----------------------------

    def __shop_has_item(self, tile: Shop, item) -> bool:
        '''does the shop have this particular item'''
        menu = getattr(tile, "shop_items", None)
        return item in menu


    def __buyable_cost(self, item: Buyable) -> int:
        '''get the cost internally'''
        # both FoodType and ShopCosts enums have .buy_cost so we can get that attr
        return int(getattr(item, "buy_cost"))

    def __grant_buyable_to_bot(self, bot_id: int, item: Buyable) -> bool:
        '''assign the purchased item to bot.holding. Returns False if unsupported'''
        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        
        if b.holding is not None:
            #self.__warn(f'buy() failed: bot {bot_id} needs to be holding nothing to buy')
            return False

        if isinstance(item, FoodType):
            b.holding = Food(item)
            return True

        if isinstance(item, ShopCosts):
            if item == ShopCosts.PLATE:
                b.holding = Plate(food=[], dirty=False)
                return True
            if item == ShopCosts.PAN:
                b.holding = Pan(None)
                return True
            #self.__warn(f"buy() failed: no shop item {item}")
            return False

        #self.__warn(f"buy() failed: no item type {type(item).__name__}")
        return False


    def can_buy(self, bot_id: int, item: Buyable, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''checks if we can buy an item that targets shop at target x, y'''
        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False

        tgt = self.__resolve_target_tile(bot_id, "can_buy()", target_x, target_y)
        if tgt is None:
            return False
        _, _, tile = tgt

        if not isinstance(tile, Shop):
            return False
        if b.holding is not None:
            return False

        # shop menu check
        if not self.__shop_has_item(tile, item):
            return False

        #check for sufficient funds
        cost = self.__buyable_cost(item)
        return self.__game_state.get_team_money(self.__team) >= cost



    def buy(self, bot_id: int, item: Buyable, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''buys the item; bot needs to not be holding anything'''

        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False

        tgt = self.__resolve_target_tile(bot_id, "buy()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        if not isinstance(tile, Shop):
            #self.__warn(f"buy() failed: target ({target_x},{target_y}) is not a shop tile for bot {bot_id}")
            return False
        if b.holding is not None:
            #self.__warn(f"buy() failed: bot {bot_id} must not carry anything when buying")
            return False

        # enforce shop menu if present
        if not self.__shop_has_item(tile, item):
            name = getattr(item, "food_name", getattr(item, "item_name", str(item)))
            #self.__warn(f"buy() failed: {name} not in shop menu")
            return False

        cost = self.__buyable_cost(item)
        if self.__game_state.get_team_money(self.__team) < cost:
            name = getattr(item, "food_name", getattr(item, "item_name", str(item)))
            #self.__warn(f"buy() failed: team {self.__team.name} insufficient funds for {name}")
            return False

        # spend money
        self.__game_state.add_team_money(self.__team, -cost)

        # give the item to the bot
        if not self.__grant_buyable_to_bot(bot_id, item):
            # if grant fails, refund the money
            self.__game_state.add_team_money(self.__team, cost)
            return False

        return True


    # ----------------------------
    # Food processing
    # ----------------------------

    def chop(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''chop on a counter'''

        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False

        tgt = self.__resolve_target_tile(bot_id, "chop()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        if not isinstance(tile, Counter):
            #self.__warn(f"chop() failed: target ({target_x},{target_y}) must be COUNTER for bot {bot_id}")
            return False
        
        if b.holding is not None:
            #self.__warn(f"chop() failed: bot {bot_id} must be holding nothing")
            return False

        item = getattr(tile, "item", None)
        if isinstance(item, Food):
            if not item.can_chop:
                #self.__warn(f"chop() failed: tile food not choppable bot {bot_id}")
                return False
            item.chopped = True
            return True

        #self.__warn(f"chop() failed: nothing choppable at ({target_x},{target_y}) for bot {bot_id}")
        return False

    def can_start_cook(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''could bot start the cook'''

        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        
        tgt = self.__resolve_target_tile(bot_id, "can_start_cook()", target_x, target_y)

        if tgt is None:
            return False
        
        _, _, tile = tgt

        if not isinstance(tile, Cooker):
            return False
        
        pan = tile.item

        if not isinstance(pan, Pan) or pan.food is not None:
            return False
        
        return isinstance(b.holding, Food) and b.holding.can_cook

    def start_cook(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''start cooking (ticks are environmental)'''

        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False

        tgt = self.__resolve_target_tile(bot_id, "start_cook()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        if not isinstance(tile, Cooker):
            #self.__warn(f"start_cook() failed: target ({target_x},{target_y}) must be cooker tile for bot {bot_id}")
            return False
        
        pan = tile.item
        if not isinstance(pan, Pan):
            #self.__warn(f"start_cook() failed: cooker at ({target_x},{target_y}) is missing pan for bot {bot_id}")
            return False
        
        if pan.food is not None:
            #self.__warn(f"start_cook() failed: pan already occupied at ({target_x},{target_y}) bot {bot_id}")
            return False
        if not (isinstance(b.holding, Food) and b.holding.can_cook):
            #self.__warn(f"start_cook() failed: bot={bot_id} must hold cookable food")
            return False

        pan.food = b.holding
        b.holding = None

        #when put the cook back on, start at the BEGINNING of the LAST stage
        if pan.food.cooked_stage == 0:
            tile.cook_progress = 0
        elif pan.food.cooked_stage == 1:
            tile.cook_progress = GameConstants.COOK_PROGRESS
        else: 
            tile.cook_progress = GameConstants.BURN_PROGRESS

        return True

    def take_from_pan(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''take food from the pan'''

        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False
        if b.holding is not None:
            #self.__warn(f"take_from_pan(): bot={bot_id} already holding something")
            return False

        tgt = self.__resolve_target_tile(bot_id, "take_from_pan()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        if not isinstance(tile, Cooker):
            #self.__warn(f"take_from_pan(): target ({target_x},{target_y}) must be COOKER bot={bot_id}")
            return False
        pan = tile.item
        if not isinstance(pan, Pan) or pan.food is None:
            #self.__warn(f"take_from_pan(): nothing in pan at ({target_x},{target_y}) bot={bot_id}")
            return False

        #take the food and resest the pan
        b.holding = pan.food
        pan.food = None
        tile.cook_progress = 0

        return True

    # ----------------------------
    # Plates and sink helpers
    # ----------------------------

    def take_clean_plate(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''take a clean plate from the sink table'''

        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False
        if b.holding is not None:
            #self.__warn(f"take_clean_plate() failed: bot {bot_id} must not carry anything")
            return False

        tgt = self.__resolve_target_tile(bot_id, "take_clean_plate()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        if not isinstance(tile, SinkTable):
            #self.__warn(f"take_clean_plate() failed: target ({target_x},{target_y}) must be a sinktable for bot {bot_id}")
            return False
        if tile.num_clean_plates <= 0:
            #self.__warn(f"take_clean_plate() failed: no clean plates available for bot={bot_id}")
            return False

        tile.num_clean_plates -= 1
        b.holding = Plate(food=[], dirty=False)
        return True

    def put_dirty_plate_in_sink(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''user carry a dirty plate, put it in the sink for washing'''

        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False
        if not isinstance(b.holding, Plate) or not b.holding.dirty:
            #self.__warn(f"put_dirty_plate_in_sink() failed: bot {bot_id} isn't holding dirty plate")
            return False

        tgt = self.__resolve_target_tile(bot_id, "put_dirty_plate_in_sink()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        if not isinstance(tile, Sink):
            #self.__warn(f"put_dirty_plate_in_sink() failed: target ({target_x},{target_y}) must be a sink tile for bot {bot_id}")
            return False

        #add dirty plate to sink
        tile.num_dirty_plates += 1
        b.holding = None
        return True

    def wash_sink(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''perform washing action, action is handled at the start of next turn's environmental tick'''
        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False

        tgt = self.__resolve_target_tile(bot_id, "wash_sink()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        if not isinstance(tile, Sink):
            #self.__warn(f"wash_sink(): target ({target_x},{target_y}) must be sink tile bot {bot_id}")
            return False
        if tile.num_dirty_plates <= 0:
            #self.__warn(f"wash_sink(): no dirty plates to wash at ({target_x},{target_y}) bot {bot_id}")
            return False

        tile.using = True
        return True

    def add_food_to_plate(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''plate a food'''
        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        if not self.__consume_action(bot_id):
            return False

        tgt = self.__resolve_target_tile(bot_id, "add_food_to_plate()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        #plate if user is holidng a plate and is targetting food
        if isinstance(b.holding, Plate):
            if b.holding.dirty:
                #self.__warn(f"add_food_to_plate() failed: plate is dirty for bot {bot_id}")
                return False
            if isinstance(getattr(tile, "item", None), Food):
                food = tile.item
                b.holding.food.append(food)
                tile.item = None
                return True
            #self.__warn(f"add_food_to_plate() failed: no food from target ({target_x},{target_y}) for bot {bot_id}")
            return False

        #plate if user is holding food and is targetting plate
        if isinstance(b.holding, Food) and isinstance(getattr(tile, "item", None), Plate):
            plate = tile.item
            if plate.dirty:
                #self.__warn(f"add_food_to_plate() failed: target plate is dirty at ({target_x},{target_y}) bot {bot_id}")
                return False
            

            plate.food.append(b.holding)
            b.holding = None
            return True

        #self.__warn(f"add_food_to_plate() failed: need a plate and food for bot {bot_id} targeting ({target_x},{target_y})")
        return False

    # --------------
    # Submit logic
    # ---------------

    def can_submit(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''can we submit the plate?'''
        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False
        if not (isinstance(b.holding, Plate) and not b.holding.dirty):
            return False
        tgt = self.__resolve_target_tile(bot_id, "can_submit()", target_x, target_y)
        if tgt is None:
            return False
        _, _, tile = tgt
        return isinstance(tile, Submit)

    def submit(self, bot_id: int, target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
        '''perform the submission action'''
        if not self.__consume_action(bot_id):
            return False

        b = self.__safe_get_bot(bot_id)
        if b is None:
            return False

        tgt = self.__resolve_target_tile(bot_id, "submit()", target_x, target_y)
        if tgt is None:
            return False
        target_x, target_y, tile = tgt

        if not isinstance(tile, Submit):
            #self.__warn(f"submit() failed: target ({target_x},{target_y}) must be submit station bot {bot_id}")
            return False
        if not isinstance(b.holding, Plate) or b.holding.dirty:
            #self.__warn(f"submit() failed: bot {bot_id} must have a clean Plate")
            return False

        #let game state handle the submission logic
        succ = self.__game_state.submit_plate(bot_id, target_x, target_y)
        if not succ:
            #self.__warn(f"submit() failed: no matching order for bot {bot_id}")
            pass
        return succ

    # ----------------------------
    # Mid-game switch mechanics (for all bots on team)
    # ----------------------------

    def get_switch_info(self) -> Dict[str, Any]:
        '''provides user with information regarding the game state's switched information'''
        
        start = self.__game_state.switch_turn
        end = self.__game_state.switch_turn + self.__game_state.switch_duration - 1
        return {
            "turn": self.__game_state.turn,
            "switch_turn": start,
            "switch_duration": self.__game_state.switch_duration,
            "window_active": (start <= self.__game_state.turn <= end),
            "window_end_turn": end,
            "my_team_switched": bool(self.__game_state.switched.get(self.__team, False)),
            "enemy_team_switched": bool(self.__game_state.switched.get(self.get_enemy_team(), False)),
        }

    def can_switch_maps(self) -> bool:
        '''Can switch ANY TIME during the map'''
        info = self.get_switch_info()
        return bool(info["window_active"]) and (not info["my_team_switched"])

    def switch_maps(self) -> bool:
        '''
        if this is called during the switch window, it tps all the bots
        into the enemy map with non-interfering spawns

        Can only be called once time per game per team

        this does not consume a bot's move or action, so they can still move this turn
        '''
        if not self.can_switch_maps():
            #self.__warn("switch_maps() failed: not allowed now (outside window or already switched).")
            return False

        success = self.__game_state.request_switch(self.__team)

        if not success:
            #self.__warn("switch_maps() failed: request rejected by GameState")
            pass

        return success


    # ----------------------------
    # Internal helpers
    # ----------------------------

    def __safe_get_bot(self, bot_id: int):
        '''get bot checkers'''
        try:
            b = self.__game_state.get_bot(bot_id)
        except Exception:
            #self.__warn(f"Invalid bot_id {bot_id}")
            return None
        if b.team != self.__team:
            #self.__warn(f"Cannot control enemy bot_id {bot_id}")
            return None
        return b


    def __item_signature(self, it: Item) -> Tuple:
        '''defines "same item" in box logic; defined similarly for the submit logic in game state'''

        #food signature
        if isinstance(it, Food):
            return ("Food", it.food_name, bool(it.chopped), int(it.cooked_stage))

        #plate signature with foods on top of it
        if isinstance(it, Plate):
            foods = []
            for f in it.food:
                if isinstance(f, Food):
                    foods.append((f.food_name, bool(f.chopped), int(f.cooked_stage)))
                else:
                    foods.append((type(f).__name__,))
            return ("Plate", bool(it.dirty), tuple(foods))

        #pan signature also by the foods
        if isinstance(it, Pan):
            if it.food is None:
                return ("Pan", None)
            return ("Pan", self.__item_signature(it.food))

        #else, just the class name
        return (type(it).__name__,)


    def __warn(self, msg: str) -> None:
        '''warn string'''
        #print(f"[RC for {self.__team.name} WARN]: {msg}")

    def __can_move_internal(self, map_team: Team, x: int, y: int, dx: int, dy: int) -> bool:
        '''private helper to see if we can move by dx, dy from x, y on map_team or not'''
        new_x, new_y = x + dx, y + dy
        m = self.__game_state.get_map(map_team)

        if not m.in_bounds(new_x, new_y):
            return False
        
        if not self.__game_state.is_walkable(map_team, new_x, new_y):
            return False
        
        occ = self.__game_state.occupancy[map_team][new_x][new_y]
        if occ is not None:
            return False
        
        return True


    def __set_cook_progress_for_food(self, cooker: Cooker, food: Food) -> None:
        '''
        internal helper that sets the colking progress for food when placed
        this is because this is used on two separate functions, modularity purposes
        '''
        if food.cooked_stage == 0:
            cooker.cook_progress = 0
        elif food.cooked_stage == 1:
            cooker.cook_progress = GameConstants.COOK_PROGRESS
        else:
            cooker.cook_progress = GameConstants.BURN_PROGRESS



    def item_to_public_dict(self, it: Optional[Item]) -> Any:
        '''basically condensces info for user'''
        if it is None:
            return None
        if isinstance(it, Food):
            return {
                "type": "Food",
                "food_name": it.food_name,
                "food_id": it.food_id,
                "chopped": it.chopped,
                "cooked_stage": it.cooked_stage,
            }
        if isinstance(it, Plate):
            return {
                "type": "Plate",
                "dirty": it.dirty,
                "food": [
                    {
                        "food_name": f.food_name,
                        "food_id": f.food_id,
                        "chopped": f.chopped,
                        "cooked_stage": f.cooked_stage,
                    }
                    for f in it.food
                    if isinstance(f, Food)
                ],
            }
        if isinstance(it, Pan):
            return {"type": "Pan", "food": self.item_to_public_dict(it.food)}
        
        return {"type": type(it).__name__}
