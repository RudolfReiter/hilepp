"""Paired held-out evaluation; selection seeds never appear in this test set."""

import argparse
import json
from pathlib import Path
import numpy as np
from vehiclegym.demo import rollout, export
from vehiclegym.simulator.racing_world import SCENARIOS
from vehiclegym.planner.references import FrenetReference
from .interface import HileppEnv
from .policy import NumpyPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("docs/data"))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    training = json.loads((args.checkpoint_dir / "training.json").read_text())
    policies = [
        ("baseline", "Fixed MPP", None),
        ("initial", "Initial neural policy", NumpyPolicy(args.checkpoint_dir / "initial.npz")),
        ("learned", "Trained SAC policy", NumpyPolicy(args.checkpoint_dir / "best.npz")),
    ]
    data = {
        "kind": "hilepp",
        "dt": 0.25,
        "horizon": 3.0,
        "scenarios": [],
        "training": training,
        "test_seeds": list(range(10000, 10000 + args.seeds)),
    }
    results = []
    for scenario in SCENARIOS:
        shown = []
        for identifier, name, policy in policies:
            scores = []
            for seed in data["test_seeds"]:
                env = HileppEnv(training["variant"], max_steps=64)
                controller = (
                    (lambda obs: FrenetReference())
                    if policy is None
                    else (lambda obs: env.expand_action(policy(obs)))
                )
                run = rollout(env.world, controller, seed, scenario, name, identifier)
                results.append(
                    {"scenario": scenario, "method": identifier, "seed": seed, **run["metrics"]}
                )
                scores.append(run["metrics"]["return"])
                if seed == data["test_seeds"][0]:
                    shown.append(run)
            print(scenario, identifier, "mean return", float(np.mean(scores)), flush=True)
        data["scenarios"].append({"id": scenario, "runs": shown})
    data["evaluation"] = results
    export(data, args.output)
    (args.output / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
