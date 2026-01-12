#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evals.llm_vs_llm import run_series  # noqa: E402


def _parse_providers(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    providers = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(providers) if providers else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM vs LLM games and log JSONL.")
    parser.add_argument("--model-a", default="openai/gpt-oss-120b")
    parser.add_argument("--model-b", default="openai/gpt-oss-20b")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="-", help="Output JSONL file path or '-' for stdout.")
    parser.add_argument("--providers", help="Comma-separated provider priority list for both models.")
    parser.add_argument("--providers-a", help="Comma-separated provider priority list for model A.")
    parser.add_argument("--providers-b", help="Comma-separated provider priority list for model B.")
    args = parser.parse_args()

    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    try:
        shared = _parse_providers(args.providers)
        providers_a = _parse_providers(args.providers_a) or shared
        providers_b = _parse_providers(args.providers_b) or shared
        run_series(
            args.model_a,
            args.model_b,
            args.games,
            args.seed,
            out,
            providers_a=providers_a,
            providers_b=providers_b,
        )
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
