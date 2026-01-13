import json

from azul_agents import LLMAgent
from azul_engine import GameEngine
from azul_engine.serialization import action_to_dict, state_to_dict

SNAPSHOT_KWARGS = {
    "include_supply_contents": True,
    "include_rng": True,
    "include_round_log": True,
    "include_first_player_index": True,
}


def _extract_rationale(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return json.loads(raw).get("rationale")
    except json.JSONDecodeError:
        return None


def _log(out, payload: dict) -> None:
    if out is None:
        return
    out.write(json.dumps(payload) + "\n")
    out.flush()


def run_series(
    model_a: str,
    model_b: str,
    games: int,
    seed: int,
    out=None,
    providers_a: tuple[str, ...] | None = None,
    providers_b: tuple[str, ...] | None = None,
    on_turn=None,
) -> list[dict]:
    results = []
    models = [model_a, model_b]
    for game_idx in range(games):
        game_seed = seed + game_idx
        engine = GameEngine(seed=game_seed)
        state = engine.reset()
        agent_a = LLMAgent(model=model_a, provider_priority=providers_a)
        agent_b = LLMAgent(model=model_b, provider_priority=providers_b)
        agents = [agent_a, agent_b]
        turn = 0

        _log(
            out,
            {
                "type": "game_start",
                "game": game_idx,
                "seed": game_seed,
                "models": models,
            },
        )

        while not state.is_terminal():
            current = state.current_player
            agent = agents[current]
            snapshot = state_to_dict(state, **SNAPSHOT_KWARGS)
            scores_before = [p.score for p in state.players]

            action = agent.select_action(state)
            state = engine.step(action)
            scores_after = [p.score for p in state.players]

            _log(
                out,
                {
                    "type": "turn",
                    "game": game_idx,
                    "turn": turn,
                    "player": current,
                    "model": agent.model,
                    "state": snapshot,
                    "scores_before": scores_before,
                    "scores_after": scores_after,
                    "llm_action_id": agent.last_action_id,
                    "llm_action_desc": agent.last_action_desc,
                    "llm_fallback": agent.last_used_fallback,
                    "llm_rationale": _extract_rationale(agent.last_raw),
                    "llm_reasoning": agent.last_reasoning,
                    "action": action_to_dict(action),
                },
            )
            if on_turn:
                info = {
                    "models": models,
                    "scores": scores_after,
                    "player": current,
                    "game": game_idx,
                }
                try:
                    on_turn(info)
                except TypeError:
                    on_turn()
            turn += 1

        scores = [p.score for p in state.players]
        max_score = max(scores)
        winners = [i for i, score in enumerate(scores) if score == max_score]
        winner = winners[0] if len(winners) == 1 else None
        result = {
            "game": game_idx,
            "seed": game_seed,
            "models": models,
            "scores": scores,
            "winner": winner,
            "winners": winners,
        }
        _log(out, {"type": "game_end", **result})
        results.append(result)
    return results
