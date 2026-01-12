#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evals.arena import run_arena  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM arena matchups in parallel.")
    parser.add_argument("--models", required=True, help="Comma-separated list of model ids.")
    parser.add_argument("--games-per-pair", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic runs (omit for random).")
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--providers", help="Comma-separated provider priority list for all models.")
    args = parser.parse_args()

    run_arena(
        args.models,
        games_per_pair=args.games_per_pair,
        seed=args.seed,
        parallel=args.parallel,
        out_dir=args.out_dir,
        providers=args.providers,
        progress=True,
    )


if __name__ == "__main__":
    main()
