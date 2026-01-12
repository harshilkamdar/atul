from dataclasses import dataclass

from .agents import Agent
from .state import GameEngine, GameState


@dataclass
class GameResult:
    final_state: GameState
    scores: list[int]


def play_game(agents: list[Agent], *, seed: int | None = None) -> GameResult:
    engine = GameEngine(num_players=len(agents), seed=seed)
    state = engine.reset()
    while not state.is_terminal():
        agent = agents[state.current_player]
        # Clone to protect state from accidental mutation by agent code.
        action = agent.select_action(state.clone())
        state = engine.step(action)
    scores = [p.score for p in state.players]
    return GameResult(final_state=state, scores=scores)


def play_series(agents: list[Agent], games: int, *, seed: int | None = None) -> list[GameResult]:
    results = []
    for i in range(games):
        game_seed = None if seed is None else seed + i
        results.append(play_game(agents, seed=game_seed))
    return results
