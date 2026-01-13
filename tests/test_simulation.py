from azul_agents import FirstLegalAgent, GreedyFillAgent, RandomAgent
from azul_engine import GamePhase, GameResult, play_game, play_series


def test_play_game_reaches_terminal_state():
    agents = [RandomAgent(), FirstLegalAgent()]
    result = play_game(agents, seed=123)
    assert isinstance(result, GameResult)
    assert result.final_state.is_terminal()
    assert result.final_state.phase == GamePhase.GAME_END
    assert len(result.scores) == 2
    assert all(score >= 0 for score in result.scores)


def test_play_series_runs_multiple_games():
    agents = [RandomAgent(), GreedyFillAgent()]
    results = play_series(agents, games=3, seed=5)
    assert len(results) == 3
    # Scores should differ across games with different seeds.
    assert len({tuple(r.scores) for r in results}) >= 1
