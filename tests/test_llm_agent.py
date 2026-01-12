from azul_engine import GameEngine, LLMAgent
import azul_engine.llm_agent as llm_agent_module


def test_llm_agent_retries_then_fallback(monkeypatch):
    state = GameEngine(seed=0).reset()
    calls = {"count": 0}

    def fake_call(prompt, model, provider_priority):
        calls["count"] += 1
        return "{bad json", None, 200, None

    monkeypatch.setattr(llm_agent_module, "_call_openrouter", fake_call)
    agent = LLMAgent(model="test-model")
    action = agent.select_action(state)

    assert calls["count"] == 2
    assert agent.last_used_fallback is True
    assert action in state.legal_actions()


def test_llm_agent_retry_success(monkeypatch):
    state = GameEngine(seed=1).reset()
    calls = {"count": 0}

    def fake_call(prompt, model, provider_priority):
        calls["count"] += 1
        if calls["count"] == 1:
            return "{bad json", None, 200, None
        return '{"action_id": 0, "rationale": "ok"}', None, 200, None

    monkeypatch.setattr(llm_agent_module, "_call_openrouter", fake_call)
    agent = LLMAgent(model="test-model")
    agent.select_action(state)

    assert calls["count"] == 2
    assert agent.last_used_fallback is False
    assert agent.last_action_id == 0
    assert agent.last_action_desc is not None
