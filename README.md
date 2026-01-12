## Azul engine skeleton

This repository implements a simple Azul rules engine to support scalable oversight experiments. The engine targets determinism and ease of simulation (legal action listing, cloning, fast scoring).

### Layout
- `src/azul_engine/enums.py` – tile and phase enumerations.
- `src/azul_engine/actions.py` – move representation and constants for center/floor.
- `src/azul_engine/player.py` – player board data structures and defaults.
- `src/azul_engine/state.py` – full game rules, legal action generation, scoring, and round management.
- `src/azul_engine/agents.py` – simple agents (random, first-legal, greedy fill).

### Notes
- Supports 2–4 players with standard factory counts and tile distribution.
- Implements drafting, pattern line validation/overflow to floor, adjacency scoring, floor penalties, and end-game bonuses (rows, columns, color sets).
- First-player token is tracked via flags rather than as a tile; penalties account for it.

### Quick start
```bash
uv run python - <<'PY'
from azul_engine import GameEngine, Action, TileColor

engine = GameEngine(seed=0)
state = engine.reset()
print(f"Legal actions at start: {len(state.legal_actions())}")
PY
```
