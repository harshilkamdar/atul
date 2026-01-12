#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evals.llm_vs_mcts import play_llm_vs_llm_with_mcts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Play LLM vs LLM and compare against MCTS per turn.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-a", default="openai/gpt-oss-120b")
    parser.add_argument("--model-b", default="openai/gpt-oss-20b")
    parser.add_argument("--mcts-budget", type=float, default=1.0)
    args = parser.parse_args()

    play_llm_vs_llm_with_mcts(
        seed=args.seed,
        model_a=args.model_a,
        model_b=args.model_b,
        mcts_budget_s=args.mcts_budget,
    )


if __name__ == "__main__":
    main()
