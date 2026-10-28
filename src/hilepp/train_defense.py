"""Train a defensive SAC specialist with an explicit auxiliary training reward.

This is an additional recreation experiment, not the paper's training recipe.
Evaluation always uses the unmodified progress/rank reward from RacingWorld.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from vehiclegym.simulator.racing_world import ENVIRONMENT_VERSION
from .interface import HileppEnv
from .sac import SAC, Replay


def defense_bonus(world, info):
    """Reward lane coverage of a nearby faster challenger and actual obstruction."""
    gaps = world.state[0] - world.opponents[:, 0]
    candidates = np.flatnonzero((gaps > 4.6) & (gaps < 45))
    coverage = 0.0
    if len(candidates):
        closest = candidates[np.argmin(gaps[candidates])]
        separation = abs(world.state[1] - world.opponents[closest, 1])
        coverage = max(0.0, 1 - separation / 4.2)
    return 0.75 * coverage + 0.75 * sum(info["held_up"])


def validate(agent):
    env = HileppEnv(max_steps=96)
    obs, _ = env.reset(8001, {"scenario": "blocking"})
    total = 0.0
    while True:
        obs, reward, end, trunc, _ = env.step(agent.act(obs, deterministic=True))
        total += reward
        if end or trunc:
            return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/defense"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = HileppEnv(max_steps=96)
    agent = SAC(seed=args.seed)
    with np.load(args.warm_start, allow_pickle=False) as weights:
        actor = agent.actor.state_dict()
        for name in weights.files:
            actor[name] = torch.from_numpy(weights[name].copy())
        agent.actor.load_state_dict(actor)
    agent.export(args.output / "initial.npz")
    replay, rng = Replay(seed=args.seed), np.random.default_rng(args.seed)
    best = validate(agent)
    agent.export(args.output / "best.npz")
    log = {
        "algorithm": "SAC with auxiliary defensive reward",
        "seed": args.seed,
        "variant": "I",
        "environment_version": ENVIRONMENT_VERSION,
        "warm_start": "initial.npz contains the copied general SAC actor before specialization",
        "training_task": "blocking",
        "auxiliary_reward": "0.75 * lane_coverage + 0.75 * held_up_count",
        "selection_reward": "unmodified progress/rank return",
        "validation_seeds": [8001],
        "requested_steps": args.steps,
        "episode_limit_steps": 96,
        "selected_step": 0,
        "episodes": [],
        "validation": [{"step": 0, "return": best}],
    }
    print(f"Defensive warm start validation: {best:.3f}", flush=True)
    obs, total, shaped_total = None, 0.0, 0.0
    started = perf_counter()
    for step in range(1, args.steps + 1):
        if obs is None:
            seed = int(rng.integers(0, 7000))
            obs, _ = env.reset(seed, {"scenario": "blocking"})
            total, shaped_total = 0.0, 0.0
        action = agent.act(obs)
        next_obs, reward, end, trunc, info = env.step(action)
        shaped = reward if end else reward + defense_bonus(env.world, info)
        replay.add(obs, action, shaped, next_obs, end)
        total += reward
        shaped_total += shaped
        if step >= 128:
            agent.update(replay.sample())
        if end or trunc:
            log["episodes"].append(
                {
                    "step": step,
                    "seed": seed,
                    "return": total,
                    "training_return": shaped_total,
                    "solver_failed": info["solver_failed"],
                    "collision": info["collision"],
                }
            )
            obs = None
        else:
            obs = next_obs
        if step % 1000 == 0 or step == args.steps:
            score = validate(agent)
            log["validation"].append({"step": step, "return": score})
            if score > best:
                best = score
                log["selected_step"] = step
                agent.export(args.output / "best.npz")
            agent.export(args.output / "last.npz")
            log.update(completed_steps=step, elapsed_s=perf_counter() - started)
            (args.output / "training.json").write_text(json.dumps(log, indent=2) + "\n")
            print(f"defense {step}: validation {score:.3f}, best {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
