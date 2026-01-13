import json

from azul_engine import GameEngine, LLMAgent, MCTSAgent
from azul_engine.serialization import action_to_dict


def _extract_rationale(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data.get("rationale")


def _diff_action(chosen, optimal) -> str:
    chosen_dict = action_to_dict(chosen)
    optimal_dict = action_to_dict(optimal)
    if chosen_dict == optimal_dict:
        return "match"
    fields = ("source_index", "color", "pattern_line", "take_first_player_token")
    return "; ".join(
        f"{field} {chosen_dict[field]} != {optimal_dict[field]}"
        for field in fields
        if chosen_dict[field] != optimal_dict[field]
    )


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
            "llm_action": action_to_dict(action),
            "llm_action_id": agent.last_action_id,
            "llm_action_desc": agent.last_action_desc,
            "llm_fallback": agent.last_used_fallback,
            "llm_rationale": _extract_rationale(agent.last_raw),
            "llm_reasoning": agent.last_reasoning,
            "mcts_action": action_to_dict(optimal),
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
