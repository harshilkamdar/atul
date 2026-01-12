import json
import math
import random
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


def _games_from_results(results: list[dict]) -> list[tuple[str, str, float]]:
    games = []
    for result in results:
        model_a = result["model_a"]
        model_b = result["model_b"]
        winner = result.get("winner")
        if winner == model_a:
            score_a = 1.0
        elif winner == model_b:
            score_a = 0.0
        else:
            score_a = 0.5
        games.append((model_a, model_b, score_a))
    return games


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _fit_elo(games: list[tuple[str, str, float]], models: list[str], steps: int, lr: float) -> dict[str, float]:
    idx = {m: i for i, m in enumerate(models)}
    ratings = [0.0] * len(models)
    step_scale = lr / max(1, len(games))
    for _ in range(steps):
        grads = [0.0] * len(models)
        for a, b, score_a in games:
            i = idx[a]
            j = idx[b]
            diff = ratings[i] - ratings[j]
            expected = _sigmoid(diff)
            delta = score_a - expected
            grads[i] += delta
            grads[j] -= delta
        for k in range(len(ratings)):
            ratings[k] += step_scale * grads[k]
        mean = sum(ratings) / len(ratings)
        ratings = [r - mean for r in ratings]
    scale = 400 / math.log(10)
    return {m: ratings[idx[m]] * scale for m in models}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def elo_with_error_bars(
    results: list[dict],
    *,
    bootstrap: int = 200,
    steps: int = 400,
    lr: float = 1.0,
    seed: int = 0,
) -> dict[str, dict]:
    games = _games_from_results(results)
    if not games:
        return {}
    models = sorted({m for g in games for m in g[:2]})
    base = _fit_elo(games, models, steps, lr)
    if bootstrap <= 0:
        return {m: {"elo": base[m], "lo": base[m], "hi": base[m]} for m in models}

    rng = random.Random(seed)
    samples = {m: [] for m in models}
    for _ in range(bootstrap):
        resample = [games[rng.randrange(len(games))] for _ in range(len(games))]
        fit = _fit_elo(resample, models, steps, lr)
        for m in models:
            samples[m].append(fit[m])

    stats = {}
    for m in models:
        stats[m] = {
            "elo": base[m],
            "lo": _percentile(samples[m], 0.025),
            "hi": _percentile(samples[m], 0.975),
            "n": len(samples[m]),
        }
    return stats


def _model_game_counts(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result["model_a"]] = counts.get(result["model_a"], 0) + 1
        counts[result["model_b"]] = counts.get(result["model_b"], 0) + 1
    return counts


def print_elo_table(stats: dict[str, dict], counts: dict[str, int]) -> None:
    headers = ["model", "games", "elo", "lo95", "hi95"]
    rows = [headers]
    for model, row in sorted(stats.items(), key=lambda x: x[1]["elo"], reverse=True):
        rows.append(
            [
                model,
                str(counts.get(model, 0)),
                f"{row['elo']:.1f}",
                f"{row['lo']:.1f}",
                f"{row['hi']:.1f}",
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    for row in rows:
        padded = [value.ljust(widths[i]) for i, value in enumerate(row)]
        print("  ".join(padded))


def summarize_with_elo(dir_path: str | Path, *, bootstrap: int = 200) -> None:
    results, matchups = summarize(dir_path)
    print("\nElo (bootstrap 95% CI)")
    stats = elo_with_error_bars(results, bootstrap=bootstrap)
    counts = _model_game_counts(results)
    print_elo_table(stats, counts)


if __name__ == "__main__":
    default_dir = Path(__file__).resolve().parent / "runs_test"
    summarize_with_elo(default_dir)
