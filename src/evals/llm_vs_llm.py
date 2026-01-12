import json

from azul_engine import GameEngine, LLMAgent


def _extract_rationale(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return json.loads(raw).get("rationale")
    except json.JSONDecodeError:
        return None


def _state_snapshot(state) -> dict:
    def rng_state(rng) -> dict:
        version, internal, gauss = rng.getstate()
        return {
            "version": version,
            "internal_state": list(internal),
            "gauss_next": gauss,
        }

    def board(p) -> dict:
        return {
            "score": p.score,
            "pattern_lines": [[t.value for t in line] for line in p.pattern_lines],
            "wall": [[bool(x) for x in row] for row in p.wall],
            "floor_line": [t.value for t in p.floor_line],
            "has_first_player_token": p.has_first_player_token,
        }

    return {
        "round": state.round_number,
        "phase": state.phase.value,
        "current_player": state.current_player,
        "first_player_index": state.first_player_index,
        "round_log": list(state.round_log),
        "rng_state": rng_state(state.rng),
        "supply": {
            "factories": [[t.value for t in f] for f in state.supply.factories],
            "center": [t.value for t in state.supply.center],
            "bag": [t.value for t in state.supply.bag],
            "discard": [t.value for t in state.supply.discard],
            "bag_count": len(state.supply.bag),
            "discard_count": len(state.supply.discard),
            "first_player_token_in_center": state.first_player_token_in_center,
        },
        "players": [board(p) for p in state.players],
    }


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
                "models": [model_a, model_b],
            },
        )

        while not state.is_terminal():
            current = state.current_player
            agent = agents[current]
            snapshot = _state_snapshot(state)
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
                    "action": {
                        "source_index": action.source_index,
                        "color": action.color.value,
                        "pattern_line": action.pattern_line,
                        "take_first_player_token": action.take_first_player_token,
                    },
                },
            )
            if on_turn:
                info = {
                    "models": [model_a, model_b],
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
        result = {
            "game": game_idx,
            "seed": game_seed,
            "models": [model_a, model_b],
            "scores": scores,
            "winner": winners[0] if len(winners) == 1 else None,
            "winners": winners,
        }
        _log(out, {"type": "game_end", **result})
        results.append(result)
    return results
