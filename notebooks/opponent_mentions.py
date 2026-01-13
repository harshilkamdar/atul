import argparse
import json
import re
from pathlib import Path

PHRASES = [
    "opponent",
    "opp",
    "enemy",
    "rival",
    "their",
    "them",
    "deny",
    "denies",
    "denied",
    "denying",
    "block",
    "blocks",
    "blocked",
    "blocking",
    "prevent",
    "prevents",
    "preventing",
    "disrupt",
    "disrupts",
    "disrupted",
    "disrupting",
    "interfere",
    "interferes",
    "interfering",
    "steal",
    "steals",
    "stole",
    "stealing",
    "snatch",
    "starve",
    "starves",
    "starving",
    "restrict",
    "restricts",
    "restricting",
    "limit",
    "limits",
    "limiting",
    "cut off",
    "punish",
    "punishes",
    "punishing",
    "threat",
    "threaten",
    "pressure",
    "pressuring",
    "initiative",
    "tempo",
    "first player",
    "first-player token",
    "take away",
    "take from",
]

DEFAULT_PATHS = [Path("notebooks/runs_final"), Path("notebooks/runs_final_new")]


def _compile_pattern(phrases):
    parts = []
    for phrase in phrases:
        if " " in phrase:
            parts.append(re.escape(phrase).replace("\\ ", "\\s+"))
        else:
            parts.append(re.escape(phrase))
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


def _iter_turns(paths):
    for path in paths:
        for file in Path(path).rglob("*.jsonl"):
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("type") != "turn":
                        continue
                    yield row


def opponent_rates(paths):
    pattern = _compile_pattern(PHRASES)
    stats = {}
    for row in _iter_turns(paths):
        model = row.get("model")
        if not model:
            continue
        stats.setdefault(model, {"turns": 0, "mentions": 0})
        stats[model]["turns"] += 1
        text = row.get("llm_rationale") or ""
        if pattern.search(text):
            stats[model]["mentions"] += 1

    rows = []
    for model, data in stats.items():
        turns = data["turns"]
        mentions = data["mentions"]
        rate = mentions / turns if turns else 0.0
        rows.append((rate, model, turns, mentions))
    rows.sort(reverse=True)
    return rows


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="directories with jsonl logs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = [Path(p) for p in args.paths] if args.paths else DEFAULT_PATHS
    rows = opponent_rates(paths)

    print("| Model | Turns | Opponent-Aware Mentions | Rate |")
    print("| --- | --- | --- | --- |")
    for rate, model, turns, mentions in rows:
        print(f"| {model} | {turns} | {mentions} | {rate * 100:.2f}% |")


if __name__ == "__main__":
    main()
