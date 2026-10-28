"""Train a new SAC policy and select checkpoints on separate validation seeds."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .interface import HileppEnv
from .sac import SAC, Replay
from vehiclegym.simulator.racing_world import ENVIRONMENT_VERSION


def validate(agent, variant):
    returns = []
    for i, scenario in enumerate(["overtaking", "blocking", "mixed"]):
        env = HileppEnv(variant, max_steps=96)
        obs, _ = env.reset(seed=8000 + i, options={"scenario": scenario})
        total = 0
        while True:
            obs, reward, end, trunc, _ = env.step(agent.act(obs, deterministic=True))
            total += reward
            if end or trunc:
                break
        returns.append(total)
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--variant", choices=["I", "II"], default="I")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = HileppEnv(args.variant, max_steps=96)
    agent = SAC(actions=env.action_size, seed=args.seed)
    replay = Replay(actions=env.action_size, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    log = {
        "seed": args.seed,
        "variant": args.variant,
        "requested_steps": args.steps,
        "algorithm": "SAC",
        "validation_seeds": [8000, 8001, 8002],
        "episodes": [],
        "validation": [],
        "environment_version": ENVIRONMENT_VERSION,
        "episode_limit_steps": 96,
    }
    agent.export(args.output / "initial.npz")
    best = validate(agent, args.variant)
    agent.export(args.output / "best.npz")
    log["validation"].append({"step": 0, "return": best})
    print(f"Initial validation return: {best:.3f}", flush=True)
    obs = None
    total = 0
    start = perf_counter()
    for step in range(1, args.steps + 1):
        if obs is None:
            seed = int(rng.integers(0, 7000))
            scenario = str(rng.choice(["overtaking", "blocking", "mixed"]))
            obs, _ = env.reset(seed, {"scenario": scenario})
            total = 0
        action = rng.uniform(-1, 1, env.action_size) if step <= 256 else agent.act(obs)
        next_obs, reward, end, trunc, info = env.step(action)
        replay.add(obs, action, reward, next_obs, end)
        total += reward
        if step > 256:
            agent.update(replay.sample())
        if end or trunc:
            log["episodes"].append(
                {
                    "step": step,
                    "scenario": scenario,
                    "seed": seed,
                    "return": total,
                    "solver_failed": info["solver_failed"],
                    "collision": info["collision"],
                }
            )
            obs = None
        else:
            obs = next_obs
        if step % 2000 == 0 or step == args.steps:
            score = validate(agent, args.variant)
            log["validation"].append({"step": step, "return": score})
            if score > best:
                best = score
                agent.export(args.output / "best.npz")
                log["selected_step"] = step
            agent.export(args.output / "last.npz")
            log["elapsed_s"] = perf_counter() - start
            log["completed_steps"] = step
            (args.output / "training.json").write_text(json.dumps(log, indent=2) + "\n")
            print(
                f"step {step}: validation {score:.3f}, best {best:.3f}, elapsed {log['elapsed_s']:.1f} s",
                flush=True,
            )
    log.setdefault("selected_step", 0)
    (args.output / "training.json").write_text(json.dumps(log, indent=2) + "\n")


if __name__ == "__main__":
    main()
