# Azul sandbox

This repo uses Azul as a sandbox for:
- learning LLM capability in rule-following, long-horizon planning, and state tracking
- training AlphaZero-style MCTS agents
- controlled experiments around imperfect reward signals for RL

## Install

Using uv:
```bash
uv venv
uv sync
```

Using pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Directory structure

- `src/azul_engine/` core rules engine and state serialization
- `src/azul_agents/` baseline agents and LLM integration
- `src/azul_nn/` neural features, models, and self-play training
- `src/evals/` arena runners and evaluation scripts
- `tests/` unit tests
- `notebooks/` analysis and experiments
