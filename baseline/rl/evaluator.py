"""Evaluate the released RL checkpoint; invoked by the unified paper runner.

Run as ``python -m baseline.rl.evaluator`` for an individual suite.
"""

import argparse
import json
import os

# The evaluator never opens a window. Selecting a non-interactive backend
# before importing TaskEnv also makes the script work on headless machines.
os.environ.setdefault("MPLBACKEND", "Agg")

import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.rl.policy import AttentionNet
from baseline.rl.traits import pad_policy_traits
from env.task_env import TaskEnv
from baseline.rl.config import EnvParams, TrainParams
from baseline.rl.rollout import Worker


# The released checkpoint and test instances use these feature dimensions.
EnvParams.TASKS_RANGE = (20, 20)
EnvParams.SPECIES_RANGE = (5, 5)
EnvParams.SPECIES_AGENTS_RANGE = (3, 3)
EnvParams.MAX_TIME = 100
EnvParams.TRAIT_DIM = 5
TrainParams.EMBEDDING_DIM = 128
TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM


def natural_key(path):
    """Sort env_2 before env_10 without an extra runtime dependency."""
    stem = Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return stem


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
    return torch.device(requested)


def load_policy(checkpoint_path, device):
    network = AttentionNet(
        TrainParams.AGENT_INPUT_DIM,
        TrainParams.TASK_INPUT_DIM,
        TrainParams.EMBEDDING_DIM,
    ).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 1.x did not expose the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint["best_model"])
    network.eval()
    return network, checkpoint


def evaluate_instance(path, network, device, sample, sample_count, seed, save_gif):
    index = natural_key(path)
    random.seed(seed + index)
    np.random.seed(seed + index)
    torch.manual_seed(seed + index)

    with Path(path).open("rb") as stream:
        env = pickle.load(stream)
    pad_policy_traits(env, EnvParams.TRAIT_DIM)

    # Worker constructs a throwaway random environment before we replace it
    # with the serialized paper instance. It shares the loaded policy for both
    # network arguments because no baseline rollout is used during evaluation.
    worker = Worker(0, network, network, 0, device)
    best = None
    start = time.perf_counter()

    for _ in range(sample_count):
        env.init_state()
        worker.env = env
        _, _, metrics = worker.run_episode(training=False, sample=sample, max_waiting=False)
        candidate = {name: float(np.asarray(value).reshape(-1)[0]) for name, value in metrics.items()}
        candidate["solved"] = float(candidate["success_rate"] == 1.0)
        if best is None or (candidate["solved"], -candidate["makespan"]) > (
            best["solved"],
            -best["makespan"],
        ):
            best = candidate

    elapsed = time.perf_counter() - start
    best["instance"] = index
    best["computation_time"] = elapsed

    if save_gif:
        gif_dir = Path(save_gif)
        gif_dir.mkdir(parents=True, exist_ok=True)
        env.plot_animation(str(gif_dir), index)

    return best


def summarize(frame):
    solved = frame[frame["solved"] == 1.0]
    summary = {
        "instances": int(len(frame)),
        "solved_instances": int(len(solved)),
        "success_rate": float(frame["solved"].mean()),
        "computation_time": float(frame["computation_time"].mean()),
    }
    for column in ("makespan", "waiting_time", "time_cost", "travel_dist", "efficiency"):
        summary[column] = float(solved[column].mean()) if len(solved) else None
        summary[f"{column}_std"] = float(solved[column].std(ddof=0)) if len(solved) else None
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, default=Path("RALTestSet"))
    parser.add_argument("--checkpoint", type=Path, default=Path("model/save/checkpoint.pth"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Rollouts per instance; 1 is RL(g.), 10 reproduces RL(s.10).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for repeatable decision ordering/action sampling.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N instances (useful for a smoke test).")
    parser.add_argument("--output", type=Path, help="CSV path (default: results/rl_greedy.csv or rl_sample_10.csv).")
    parser.add_argument("--save-gif", type=Path, help="Optional directory in which to render per-instance GIFs.")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    return args


def main():
    args = parse_args()
    files = sorted(args.test_set.glob("env_*.pkl"), key=natural_key)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No env_*.pkl instances found in {args.test_set}")

    device = resolve_device(args.device)
    network, checkpoint = load_policy(args.checkpoint, device)
    is_sampling = args.samples > 1
    print(
        f"Loaded {args.checkpoint} (episode {checkpoint.get('episode', 'unknown')}) on {device}; "
        f"evaluating {len(files)} instances with {args.samples} rollout(s) each."
    )

    rows = []
    for position, path in enumerate(files, start=1):
        row = evaluate_instance(
            path,
            network,
            device,
            sample=is_sampling,
            sample_count=args.samples,
            seed=args.seed,
            save_gif=args.save_gif,
        )
        rows.append(row)
        print(
            f"[{position:>2}/{len(files)}] {path.name}: solved={bool(row['solved'])} "
            f"makespan={row['makespan']:.3f} time={row['computation_time']:.3f}s"
        )

    frame = pd.DataFrame(rows).sort_values("instance")
    default_name = "rl_greedy.csv" if args.samples == 1 else f"rl_sample_{args.samples}.csv"
    output = args.output or Path("results") / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = summarize(frame)
    report = {
        "configuration": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_episode": checkpoint.get("episode"),
            "test_set": str(args.test_set),
            "device": str(device),
            "samples": args.samples,
            "seed": args.seed,
        },
        "results": summary,
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\nAggregate results")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {output} and {report_path}")


if __name__ == "__main__":
    main()
