# Carnegie Cookoff

## Init

### Create a venv

Mac/Linux

```bash
    python3 -m venv .venv
    source .venv/bin/activate
```

Windows

```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
    pip install --upgrade pip
    pip install -r requirements.txt
```



## Run

To run for results:

```bash
    python src/game.py --red bots/duo_noodle_bot.py --blue bots/duo_noodle_bot.py --map maps/map1.txt
```


To run with local pygame renderer:

```bash
    python src/game.py --red bots/duo_noodle_bot.py --blue bots/duo_noodle_bot.py --map maps/map1.txt --render
```

## Bot API Document

[API Google Doc](https://docs.google.com/document/d/1nUkWxDJRSEe4xSbe1q4rNd6GeMOpzQO-H_nWJHBnP14/edit?tab=t.0#heading=h.itwj41env6xx)


## Repo Structure

- **`src/game.py`**
  - Main entry point to the engine

- **`src/game_state.py`**

- **`src/robot_controller.py`**
  - The API that participants use to control the bots
  - Key rules:
    - each bot gets **1 move + 1 action per turn**
    - actions must target within Chebyshev distance 1
    - need correct targets

- **`src/game_constants.py`**

- **`src/map_processor.py`**

- **`src/map.py`**

- **`src/tiles.py`**

- **`src/item.py`**

- **`src/render.py`**
  - Pygame renderer helpers to visualize both maps, bots, items, and the HUD (turn, money, active orders).

- **`bots/*.py`**
  - Each bot file must define the following:
    ```python
    class BotPlayer:
        def __init__(self, map_copy): ...
        def play_turn(self, controller): ...
    ```

- **`maps/*.txt`**
    - sample maps



## Map File Format

### Tiles Legend
| Char | Tile |
|------|------|
| `.`  | Floor |
| `#`  | Wall |
| `C`  | Counter |
| `K`  | Cooker |
| `S`  | Sink |
| `T`  | SinkTable |
| `R`  | Trash |
| `U`  | Submit |
| `$`  | Shop |
| `B`  | Box |
| `b`  | Bot spawn (both teams) |

### Example
See maps/maps1.txt file