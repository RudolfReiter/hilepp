"""Matched six-car races for sampled SAC policies, actor means and pure MPC."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from vehiclegym.simulator.circuit_race import CircuitRace, PROTOCOL
from vehiclegym.race_replay import record_race, write_replay, write_manifest
from .policy import NumpyPolicy, NumpyGaussianPolicy
from .circuit import expand_action


def cached_mpc(path, world, seed, duration, track):
    run = json.loads(gzip.decompress(path.read_bytes()))
    world.reset(seed)
    if (
        run["protocol"] != PROTOCOL
        or run["method"] != "MPC"
        or run["seed"] != seed
        or run["track"]["name"] != track
        or run["planner"] != asdict(world.config)
        or run["duration_limit_s"] != duration
    ):
        raise ValueError("Cached MPC protocol does not match this evaluation")
    np.testing.assert_allclose(run["states"][0], world.states, atol=1e-6, rtol=0)
    for vehicle, profile in zip(run["vehicles"], world.profiles):
        if vehicle["reference"] != asdict(profile):
            raise ValueError("Cached opponent parameters differ")
    if [vehicle["limits"] for vehicle in run["vehicles"]] != [asdict(v) for v in world.limits]:
        raise ValueError("Cached vehicle limits differ")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", choices=["HILEPP", "SAC", "MPC"], default=["HILEPP", "SAC", "MPC"]
    )
    parser.add_argument("--execution", choices=["both", "sampled", "mean"], default="both")
    parser.add_argument("--policy-seed", type=int, default=60000)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints/deepsets"))
    parser.add_argument("--output", type=Path, default=Path("docs/data"))
    parser.add_argument(
        "--mpc-replays",
        type=Path,
        help="Reuse verified identical pure-MPC recordings from vehicle_gym",
    )
    parser.add_argument("--tracks", nargs="+", default=["Oval", "IMS", "Sepang"])
    parser.add_argument("--duration", type=float, default=180)
    parser.add_argument("--oval-duration", type=float, default=240)
    parser.add_argument("--seeds", nargs="+", type=int, default=[30000, 30001, 30002])
    args = parser.parse_args()
    entries = []
    records = []
    training = {}
    modes = ["sampled", "mean"] if args.execution == "both" else [args.execution]
    policies = {}
    for mode in modes:
        cls = NumpyGaussianPolicy if mode == "sampled" else NumpyPolicy
        suffix = "-distribution.npz" if mode == "sampled" else ".npz"
        policies[mode] = {}
        for method in args.methods:
            if method == "MPC":
                continue
            metadata = json.loads((args.checkpoints / (method.lower() + ".json")).read_text())
            if (
                metadata.get("architecture") != "deep-sets-sum-v1"
                or metadata["protocol"] != PROTOCOL
            ):
                raise ValueError(
                    "Circuit evaluation requires a freshly trained Deep Sets checkpoint"
                )
            checkpoint = args.checkpoints / (method.lower() + suffix)
            key = "distribution_sha256" if mode == "sampled" else "checkpoint_sha256"
            if metadata[key] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                raise ValueError("Checkpoint and training manifest disagree")
            if method == "HILEPP" and metadata.get("speed_reference_range") != [20, 28.0]:
                raise ValueError(
                    "Use the final-stage HILEPP checkpoint with a 20–28 m/s reference range"
                )
            policies[mode][method] = cls(checkpoint)
            if not policies[mode][method].deep_sets:
                raise ValueError("The exported actor does not contain a Deep Sets encoder")
            training[method] = {key: value for key, value in metadata.items() if key != "episodes"}
    for track in args.tracks:
        duration = args.oval_duration if track == "Oval" else args.duration
        world = CircuitRace(track, duration)
        if any(t["vehicle_limits"] != [asdict(v) for v in world.limits] for t in training.values()):
            raise ValueError("Training and evaluation vehicle limits differ")
        for seed in args.seeds:
            baseline = None
            for mode in modes:
                for method in args.methods:
                    filename = f"{track.lower()}-{method.lower()}-{seed}.json.gz"
                    if method == "MPC":
                        if baseline is None:
                            baseline = (
                                cached_mpc(
                                    args.mpc_replays / filename, world, seed, duration, track
                                )
                                if args.mpc_replays
                                else record_race(world, seed, progress=True)
                            )
                        run = baseline
                        relative = filename
                    else:
                        policy = policies[mode][method]
                        if mode == "sampled":
                            policy.reset_rng(args.policy_seed + seed)
                        run = record_race(
                            world,
                            seed,
                            method,
                            lambda obs, world: expand_action(policy(obs), method),
                            progress=True,
                        )
                        run["execution"] = dict(
                            mode=mode,
                            policy_seed=args.policy_seed + seed if mode == "sampled" else None,
                        )
                        relative = f"{mode}/{filename}"
                    write_replay(run, args.output / relative)
                    records.append(
                        dict(
                            method=method,
                            seed=seed,
                            track=track,
                            execution=mode,
                            metrics=run["metrics"],
                        )
                    )
                    entries.append(
                        dict(
                            id=f"{mode}-{track.lower()}-{method.lower()}-{seed}",
                            method=method,
                            seed=seed,
                            track=track,
                            execution=mode,
                            file=relative,
                            metrics=run["metrics"],
                        )
                    )
                    print(track, mode, method, seed, run["metrics"], flush=True)
    write_manifest(args.output, "hilepp", entries, records, training)


if __name__ == "__main__":
    main()
