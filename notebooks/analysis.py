import json
from pathlib import Path


def load_game_results(dir_path: str | Path) -> list[dict]:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"missing dir: {dir_path}")
    results = []
    for path in sorted(dir_path.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("type") != "game_end":
                    continue
                models = row.get("models") or []
                if len(models) != 2:
                    continue
                winner = row.get("winner")
                winners = row.get("winners") or []
                if winner is None and len(winners) == 1:
                    winner = winners[0]
                winner_model = models[winner] if winner in (0, 1) else None
                results.append(
                    {
                        "model_a": models[0],
                        "model_b": models[1],
                        "scores": row.get("scores"),
                        "winner": winner_model,
                        "file": str(path),
                        "game": row.get("game"),
                    }
                )
    return results


def aggregate_matchups(results: list[dict]) -> dict:
    matchups: dict[tuple[str, str], dict] = {}
    for result in results:
        model_a = result["model_a"]
        model_b = result["model_b"]
        key = tuple(sorted([model_a, model_b]))
        entry = matchups.setdefault(key, {"wins": {}, "draws": 0, "total": 0})
        wins = entry["wins"]
        wins.setdefault(model_a, 0)
        wins.setdefault(model_b, 0)
        entry["total"] += 1
        if result["winner"] is None:
            entry["draws"] += 1
        else:
            wins[result["winner"]] += 1
    return matchups


def print_table(matchups: dict) -> None:
    headers = ["model_a", "model_b", "total", "wins_a", "wins_b", "draws", "wr_a", "wr_b"]
    rows = [headers]
    for (model_a, model_b), entry in sorted(matchups.items()):
        total = entry["total"]
        wins_a = entry["wins"].get(model_a, 0)
        wins_b = entry["wins"].get(model_b, 0)
        draws = entry["draws"]
        wr_a = wins_a / total if total else 0.0
        wr_b = wins_b / total if total else 0.0
        rows.append(
            [
                model_a,
                model_b,
                str(total),
                str(wins_a),
                str(wins_b),
                str(draws),
                f"{wr_a:.2f}",
                f"{wr_b:.2f}",
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    for row in rows:
        padded = [value.ljust(widths[i]) for i, value in enumerate(row)]
        print("  ".join(padded))


def summarize(dir_path: str | Path) -> tuple[list[dict], dict]:
    results = load_game_results(dir_path)
    matchups = aggregate_matchups(results)
    print_table(matchups)
    return results, matchups


if __name__ == "__main__":
    default_dir = Path(__file__).resolve().parent / "runs_test"
    summarize(default_dir)
