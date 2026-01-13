import argparse
import json
import os
import random
import time

import torch

from .alphazero import LossWeights, ReplayBuffer, SelfPlayConfig, generate_self_play, train_epoch
from .model import AzulNet


def _write_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as out:
        out.write(json.dumps(payload) + "\n")


def _mean(metrics, key):
    values = [m[key] for m in metrics if key in m]
    return sum(values) / len(values) if values else None


def train_loop(
    *,
    out_dir: str = "runs/train",
    iterations: int = 1000,
    games_per_iter: int = 20,
    buffer_capacity: int = 50_000,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int | None = None,
    num_simulations: int = 100,
    cpuct: float = 0.5,
    temperature: float = 1.0,
    temperature_end: float = 0.1,
    temperature_moves: int = 8,
    dirichlet_alpha: float | None = 0.3,
    dirichlet_epsilon: float = 0.25,
    checkpoint_every: int = 10,
    device: str | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.jsonl")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    rng = random.Random(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    net = AzulNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    weights = LossWeights()

    cfg = SelfPlayConfig(
        num_simulations=num_simulations,
        cpuct=cpuct,
        temperature=temperature,
        temperature_end=temperature_end,
        temperature_moves=temperature_moves,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )

    meta_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as out:
            json.dump(
                {
                    "iterations": iterations,
                    "games_per_iter": games_per_iter,
                    "buffer_capacity": buffer_capacity,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                    "device": str(device),
                    "self_play": cfg.__dict__,
                },
                out,
                indent=2,
            )

    for iteration in range(iterations):
        t0 = time.perf_counter()
        game_seed = rng.randrange(1_000_000_000)
        net.eval()
        samples = generate_self_play(net, cfg, games_per_iter, device=device, seed=game_seed)
        buffer.add_samples(samples)
        net.train()
        metrics = train_epoch(net, optimizer, buffer, batch_size=batch_size, device=device, weights=weights)
        elapsed = time.perf_counter() - t0

        log = {
            "iter": iteration,
            "games": games_per_iter,
            "samples": len(samples),
            "buffer": len(buffer),
            "seconds": elapsed,
            "loss": _mean(metrics, "loss"),
            "policy_loss": _mean(metrics, "policy_loss"),
            "win_loss": _mean(metrics, "win_loss"),
            "margin_loss": _mean(metrics, "margin_loss"),
            "score_loss": _mean(metrics, "score_loss"),
        }
        _write_jsonl(log_path, log)

        if checkpoint_every and (iteration + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{iteration + 1:05d}.pt")
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration + 1,
                    "config": cfg.__dict__,
                },
                ckpt_path,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/train")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--cpuct", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.1)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_loop(
        out_dir=args.out_dir,
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_simulations=args.num_simulations,
        cpuct=args.cpuct,
        temperature=args.temperature,
        temperature_end=args.temperature_end,
        temperature_moves=args.temperature_moves,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
