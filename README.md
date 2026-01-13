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

## LLM arena snapshot

Computed from completed games in `notebooks/runs_final` and `notebooks/runs_final_new`.
Win rate counts draws as 0.5 win.

| Model | Elo | Games | W | L | D | Win Rate | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| google/gemini-3-flash-preview | 152.2 | 67 | 46 | 18 | 3 | 70.9% | 83.3–248.4 |
| openai/gpt-5-mini | 58.1 | 67 | 35 | 28 | 4 | 55.2% | -27.4–154.4 |
| x-ai/grok-4.1-fast | 44.2 | 48 | 22 | 22 | 4 | 50.0% | -68.8–141.2 |
| openai/gpt-oss-120b | -79.4 | 66 | 17 | 46 | 3 | 28.0% | -186.9–37.9 |
| anthropic/claude-haiku-4.5 | -175.1 | 10 | 2 | 8 | 0 | 20.0% | -464.1–9.3 |
