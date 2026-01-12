import json

from azul_engine import GameEngine, LLMAgent, MCTSAgent


def _extract_rationale(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data.get("rationale")


def _action_signature(action) -> tuple:
    return (
        action.source_index,
        action.color,
        action.pattern_line,
        action.take_first_player_token,
    )


def _diff_action(chosen, optimal) -> str:
    if _action_signature(chosen) == _action_signature(optimal):
        return "match"
    diffs = []
    if chosen.source_index != optimal.source_index:
        diffs.append(f"source_index {chosen.source_index} != {optimal.source_index}")
    if chosen.color != optimal.color:
        diffs.append(f"color {chosen.color} != {optimal.color}")
    if chosen.pattern_line != optimal.pattern_line:
        diffs.append(f"pattern_line {chosen.pattern_line} != {optimal.pattern_line}")
    if chosen.take_first_player_token != optimal.take_first_player_token:
        diffs.append(
            "take_first_player_token "
            f"{chosen.take_first_player_token} != {optimal.take_first_player_token}"
        )
    return "; ".join(diffs)


def play_llm_vs_llm_with_mcts(
    seed: int,
    model_a: str,
    model_b: str,
    mcts_budget_s: float = 1.0,
    log: bool = True,
) -> dict:
    engine = GameEngine(seed=seed)
    state = engine.reset()
    agents = [LLMAgent(model=model_a), LLMAgent(model=model_b)]
    mcts = MCTSAgent(time_budget_s=mcts_budget_s)
    turn = 0
    turns = []

    while not state.is_terminal():
        current = state.current_player
        agent = agents[current]

        action = agent.select_action(state)
        optimal = mcts.select_action(state.clone())

        entry = {
            "turn": turn,
            "player": current,
            "model": agent.model,
            "scores_before": [p.score for p in state.players],
            "llm_action": _action_signature(action),
            "llm_action_id": agent.last_action_id,
            "llm_action_desc": agent.last_action_desc,
            "llm_fallback": agent.last_used_fallback,
            "llm_rationale": _extract_rationale(agent.last_raw),
            "llm_reasoning": agent.last_reasoning,
            "mcts_action": _action_signature(optimal),
            "diff": _diff_action(action, optimal),
        }
        turns.append(entry)

        if log:
            print(f"TURN {turn} | player {current}")
            print(state)
            print("scores:", entry["scores_before"])
            print("llm_action:", action)
            print("llm_rationale:", entry["llm_rationale"])
            if entry["llm_reasoning"]:
                print("llm_reasoning:", entry["llm_reasoning"])
            print("mcts_action:", optimal)
            print("diff:", entry["diff"])

        state = engine.step(action)
        turn += 1

    scores = [p.score for p in state.players]
    if log:
        print("final_scores:", scores)
    return {"scores": scores, "turns": turns}
