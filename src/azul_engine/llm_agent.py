"""LLM agent that picks Azul moves via OpenRouter."""

import json
import logging
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass

from .actions import Action
from .agents import prune_and_order_actions
from .state import GameState
from .serialization import state_to_dict


DEFAULT_MODEL = "google/gemini-3-pro-preview"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
LOGGER = logging.getLogger(__name__)
RULES_TEXT = """You are an expert Azul player. Given the full game state, choose the best legal move for the current player.
Rules summary (standard wall):
- On your turn: pick one source (factory or center), take ALL tiles of ONE color; remaining tiles from a factory go to center.
- Place all taken tiles in exactly one pattern line or floor. A pattern line can only hold one color and cannot exceed its capacity.
- Cannot place color X in a row whose wall already has color X.
- Overflow goes to floor; floor penalties: -1, -1, -2, -2, -2, -3, -3 (score floors at 0).
- Wall tiling scores: isolated tile = 1; otherwise horizontal run length + vertical run length (both if present).
- Endgame bonuses: +2 per complete row, +7 per complete column, +10 per complete color set.
- Game ends after a row is completed; finish the round and apply bonuses.
Input schema:
- State is JSON with round, phase, current_player, supply (factories, center, bag_count, discard_count, first_player_token_in_center), and players:
  - Each player has score, pattern_lines (lists of colors per line), wall (5x5 booleans), floor_line (colors), has_first_player_token.
- Legal moves: list of indexed descriptions. Each move moves all tiles of one color from a source (factory/center) to one pattern line (1-5) or floor; take_first_player_token indicates grabbing the token from center.
Output schema:
Return JSON only: {"action_id": <int>, "rationale": "<brief>"}.
"""


def _format_action(idx: int, action: Action) -> str:
    src = "center" if action.source_index == Action.CENTER else f"factory {action.source_index}"
    dest = "floor" if action.pattern_line == Action.FLOOR else f"pattern line {action.pattern_line + 1}"
    token = " (takes first player token)" if action.take_first_player_token else ""
    return f"{idx}: take {action.color.value} from {src} to {dest}{token}"


def _render_prompt(state: GameState, actions: list[Action]) -> str:
    state_block = state_to_dict(state)

    lines = [
        RULES_TEXT,
        "State:",
        json.dumps(state_block, indent=2),
        "Legal moves (index -> description):",
    ]
    for idx, action in enumerate(actions):
        lines.append(_format_action(idx, action))
    lines.append('Respond with JSON: {"action_id": <int>, "rationale": "<short reason>"}')
    return "\n".join(lines)


def _call_openrouter(prompt: str, model: str, provider_priority: tuple[str, ...] | None):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": True, "effort": "medium"},
        "allow_fallbacks": False,
    }
    if provider_priority:
        body["provider"] = {"order": list(provider_priority)}
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_API_BASE,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
        data = json.loads(body)
        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        # OpenRouter returns top-level 'reasoning' or message-level 'reasoning_content'
        reasoning = choice.get("reasoning") or message.get("reasoning_content")
        status = 200
    except urllib.error.HTTPError as e:
        return None, None, e.code, e.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return None, None, None, str(e)
    return content, reasoning, status, None


def _extract_action_id(text: str) -> tuple[int | None, str | None]:
    if not text:
        return None, "empty_response"
    cleaned = text.strip()
    brace_idx = cleaned.find("{")
    bracket_idx = cleaned.find("[")
    starts = [idx for idx in (brace_idx, bracket_idx) if idx != -1]
    if not starts:
        return None, "no_json_start"
    start = min(starts)
    try:
        payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None, "json_decode_error"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    if "action_id" not in payload:
        return None, "missing_action_id"
    try:
        return int(payload["action_id"]), None
    except (TypeError, ValueError):
        return None, "invalid_action_id"


@dataclass
class LLMAgent:
    model: str = DEFAULT_MODEL
    last_raw: str | None = None
    last_reasoning: str | None = None
    provider_priority: tuple[str, ...] | None = ("fireworks",)
    last_status: int | None = None
    last_error: str | None = None
    last_failure_reason: str | None = None
    last_action_id: int | None = None
    last_action_desc: str | None = None
    last_used_fallback: bool = False

    def select_action(self, state: GameState) -> Action:
        actions = prune_and_order_actions(state, state.legal_actions(), top_k=None)
        if not actions:
            raise RuntimeError("no legal actions available")
        prompt = _render_prompt(state, actions)
        self.last_action_id = None
        self.last_action_desc = None
        self.last_used_fallback = False
        self.last_failure_reason = None
        for attempt in range(2):
            try:
                content, reasoning, status, error = _call_openrouter(prompt, self.model, self.provider_priority)
                self.last_raw = content
                self.last_reasoning = reasoning
                self.last_status = status
                self.last_error = error
                if content:
                    action_id, failure = _extract_action_id(content)
                    if action_id is not None and 0 <= action_id < len(actions):
                        self.last_action_id = action_id
                        self.last_action_desc = _format_action(action_id, actions[action_id])
                        self.last_failure_reason = None
                        return actions[action_id]
                    self.last_failure_reason = (
                        "action_id_out_of_range" if action_id is not None else failure
                    )
            except Exception as exc:
                self.last_raw = None
                self.last_reasoning = None
                self.last_error = str(exc)
                self.last_status = None
                self.last_failure_reason = "call_exception"
        # Fallback: pick a random legal action and warn.
        self.last_used_fallback = True
        snippet = (self.last_raw or "").replace("\n", " ")
        if len(snippet) > 200:
            snippet = f"{snippet[:200]}..."
        LOGGER.warning(
            "LLMAgent fallback to random action after 2 attempts "
            "(model=%s status=%s error=%s reason=%s raw=%s)",
            self.model,
            self.last_status,
            self.last_error,
            self.last_failure_reason,
            snippet or None,
        )
        chosen = random.choice(actions)
        action_id = actions.index(chosen)
        self.last_action_id = action_id
        self.last_action_desc = _format_action(action_id, chosen)
        return chosen

    def render_prompt(self, state: GameState) -> str:
        """Expose prompt for debugging without making network calls."""
        actions = prune_and_order_actions(state, state.legal_actions(), top_k=None)
        if not actions:
            raise RuntimeError("no legal actions available")
        return _render_prompt(state, actions)
