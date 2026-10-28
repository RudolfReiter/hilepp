"""Paired held-out evaluation; selection seeds never appear in this test set."""

import argparse
import json
from pathlib import Path
import numpy as np
from vehiclegym.demo import rollout, export
from vehiclegym.simulator.racing_world import SCENARIOS, ENVIRONMENT_VERSION
from vehiclegym.planner.references import FrenetReference
from .interface import HileppEnv
from .policy import NumpyPolicy, limit_lateral_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("docs/data"))
    parser.add_argument("--defense-dir", type=Path, default=None)
    parser.add_argument(
        "--defense-reference-limit",
        type=float,
        default=None,
        help="Add a diagnostic defensive run with this maximum lateral reference offset (m)",
    )
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.defense_reference_limit is not None and (
        args.defense_dir is None or args.defense_reference_limit <= 0
    ):
        parser.error("A positive reference limit requires --defense-dir")
    training = json.loads((args.checkpoint_dir / "training.json").read_text())
    if training.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError("Checkpoint was trained with a different scenario/prediction protocol")
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
        "environment_version": ENVIRONMENT_VERSION,
    }
    results = []
    for scenario in SCENARIOS:
        shown = []
        for identifier, name, policy in policies:
            scores = []
            for seed in data["test_seeds"]:
                env = HileppEnv(training["variant"], max_steps=96)
                controller = (
                    (lambda obs: FrenetReference(speed=32))
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
    if args.defense_dir is not None:
        defense_training = json.loads((args.defense_dir / "training.json").read_text())
        if defense_training.get("environment_version") != ENVIRONMENT_VERSION:
            raise ValueError("Defensive checkpoint uses a different scenario protocol")
        data["defense_training"] = defense_training
        policy = NumpyPolicy(args.defense_dir / "best.npz")
        for seed in data["test_seeds"]:
            env = HileppEnv("I", max_steps=96)
            run = rollout(
                env.world,
                lambda obs: env.expand_action(policy(obs)),
                seed,
                "blocking",
                "Defensive SAC · auxiliary reward",
                "defense",
            )
            results.append(
                {"scenario": "blocking", "method": "defense", "seed": seed, **run["metrics"]}
            )
            print("blocking defense", seed, run["metrics"], flush=True)
            if seed == data["test_seeds"][0]:
                next(s for s in data["scenarios"] if s["id"] == "blocking")["runs"].append(run)
        if args.defense_reference_limit is not None:
            data["defense_reference_limit"] = args.defense_reference_limit
            for seed in data["test_seeds"]:
                env = HileppEnv("I", max_steps=96)
                raw_references = []

                def limited(obs):
                    reference = env.expand_action(policy(obs))
                    raw_references.append(reference.lateral)
                    return limit_lateral_reference(
                        reference, env.world.state[1], args.defense_reference_limit
                    )

                run = rollout(
                    env.world,
                    limited,
                    seed,
                    "blocking",
                    "Defensive SAC · limited reference",
                    "defense_limited",
                )
                run["raw_lateral_references"] = raw_references
                run["reference_limit_m"] = args.defense_reference_limit
                results.append(
                    {
                        "scenario": "blocking",
                        "method": "defense_limited",
                        "seed": seed,
                        **run["metrics"],
                    }
                )
                print("blocking defense_limited", seed, run["metrics"], flush=True)
                if seed == data["test_seeds"][0]:
                    next(s for s in data["scenarios"] if s["id"] == "blocking")["runs"].append(run)
    data["evaluation"] = results
    export(data, args.output)
    (args.output / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
