import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .llm_vs_llm import run_series

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional
    tqdm = None


@dataclass(frozen=True)
class Task:
    model_a: str
    model_b: str
    run_idx: int
    seed: int
    out_path: Path


def _parse_models(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def _parse_providers(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    providers = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(providers) if providers else None


def _coerce_provider_list(raw: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return _parse_providers(raw)
    if isinstance(raw, tuple):
        return raw if raw else None
    if isinstance(raw, list):
        if not raw:
            return None
        if not all(isinstance(p, str) for p in raw):
            raise TypeError("provider list must be strings")
        return tuple(p.strip() for p in raw if p and p.strip()) or None
    raise TypeError("unsupported providers spec")


def _normalize_provider_map(
    models: list[str],
    providers,
) -> dict[str, tuple[str, ...] | None]:
    if providers is None:
        return {m: None for m in models}
    if isinstance(providers, dict):
        extras = set(providers) - set(models)
        if extras:
            extras_list = ", ".join(sorted(extras))
            raise ValueError(f"unknown models in providers map: {extras_list}")
        return {m: _coerce_provider_list(providers.get(m)) for m in models}
    if isinstance(providers, list) and providers and not all(isinstance(p, str) for p in providers):
        if len(providers) != len(models):
            raise ValueError("providers list must match models length")
        return {m: _coerce_provider_list(p) for m, p in zip(models, providers)}
    shared = _coerce_provider_list(providers)
    return {m: shared for m in models}


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _build_tasks(
    models: list[str],
    games_per_pair: int,
    seed: int | None,
    out_dir: Path,
    swap: bool,
) -> list[Task]:
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    used_seeds = set()
    tasks = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            pair_slug = f"{_slug(model_a)}_vs_{_slug(model_b)}"
            swap_flags = [True] * (games_per_pair // 2)
            swap_flags.extend([False] * (games_per_pair - len(swap_flags)))
            if swap:
                rng.shuffle(swap_flags)
            for run_idx, swap_sides in enumerate(swap_flags):
                play_a, play_b = (model_b, model_a) if (swap and swap_sides) else (model_a, model_b)
                while True:
                    game_seed = rng.randrange(1_000_000_000)
                    if game_seed not in used_seeds:
                        used_seeds.add(game_seed)
                        break
                out_path = out_dir / f"{pair_slug}_{run_idx:03d}.jsonl"
                tasks.append(Task(play_a, play_b, run_idx, game_seed, out_path))
    rng.shuffle(tasks)
    return tasks


def _run_task(
    task: Task,
    provider_map: dict[str, tuple[str, ...] | None],
    on_turn,
) -> None:
    os.makedirs(task.out_path.parent, exist_ok=True)
    with task.out_path.open("w", encoding="utf-8") as out:
        run_series(
            task.model_a,
            task.model_b,
            games=1,
            seed=task.seed,
            out=out,
            providers_a=provider_map.get(task.model_a),
            providers_b=provider_map.get(task.model_b),
            on_turn=on_turn,
        )


def run_arena(
    models: list[str] | str,
    *,
    games_per_pair: int = 40,
    seed: int | None = None,
    parallel: int = 16,
    out_dir: str | Path = "runs",
    swap_sides: bool = True,
    providers=None,
    progress: bool = True,
) -> list[Path]:
    if isinstance(models, str):
        models = _parse_models(models)
    if len(models) < 2:
        raise ValueError("need at least two models")

    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    tasks = _build_tasks(models, games_per_pair, seed, out_dir, swap_sides)

    provider_map = _normalize_provider_map(models, providers)

    total = len(tasks)
    max_workers = min(parallel, total) if total else 0
    if max_workers == 0:
        return [task.out_path for task in tasks]

    tqdm_func = tqdm if callable(tqdm) else None
    turn_bar = None
    turn_lock = None
    on_turn = None
    if progress and tqdm_func is not None:
        turn_lock = threading.Lock()
        turn_bar = tqdm_func(total=None, desc="turns", position=1, leave=False)

        def on_turn(info) -> None:
            model_a, model_b = info["models"]
            score_a, score_b = info["scores"]
            def short(name: str, limit: int = 24) -> str:
                return name if len(name) <= limit else f"{name[:limit-3]}..."

            label = f"g{info['game']} {short(model_a)} vs {short(model_b)} | {score_a}-{score_b}"
            with turn_lock:
                turn_bar.set_postfix_str(label)
                turn_bar.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_task, task, provider_map, on_turn): task for task in tasks}
        if progress and tqdm_func is not None:
            for future in tqdm_func(as_completed(futures), total=total, desc="games", position=0):
                future.result()
        else:
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if progress:
                    print(f"{completed}/{total} complete")

    if turn_bar is not None:
        turn_bar.close()

    return [task.out_path for task in tasks]
