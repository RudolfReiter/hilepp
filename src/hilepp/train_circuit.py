"""Train fresh SAC actors; no scripted maneuvers, imitation data or MPC fallback."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from vehiclegym.simulator.circuit_race import CircuitRace, PROTOCOL
from .circuit import expand_action
from .sac import SAC, Replay
from .encoders import ARCHITECTURE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["HILEPP", "SAC"], required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--tracks", nargs="+", default=["Oval", "IMS"])
    parser.add_argument("--episode-duration", type=float, default=90.0)
    parser.add_argument("--speed-ceiling", type=float, default=28.0)
    parser.add_argument(
        "--initialize",
        type=Path,
        help="Directory containing locally generated full training checkpoints",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/deepsets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    actions = 4 if args.method == "HILEPP" else 2
    agent = SAC(observations=31, actions=actions, seed=args.seed, entropy=0.03, deep_sets=True)
    replay = Replay(
        observations=31, actions=actions, capacity=max(50000, args.steps), seed=args.seed
    )
    prior = None
    if args.initialize:
        prefix = args.initialize / args.method.lower()
        prior = json.loads(prefix.with_suffix(".json").read_text())
        checkpoint = torch.load(str(prefix) + "-training.pt", map_location="cpu", weights_only=True)
        if (
            checkpoint["steps"] != prior["steps"]
            or prior["method"] != args.method
            or prior["observations"] != 31
            or prior.get("architecture") != ARCHITECTURE
            or prior.get("protocol") != PROTOCOL
        ):
            raise ValueError("Initialization checkpoint and manifest do not match")
        for name in ["actor", "critics", "targets"]:
            getattr(agent, name).load_state_dict(checkpoint[name])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    previous_steps = prior["steps"] if prior else 0
    previous_updates = prior["gradient_updates"] if prior else 0
    worlds = [CircuitRace(track=track, duration=args.episode_duration) for track in args.tracks]
    world = worlds[0]
    rng = np.random.default_rng(args.seed)
    episode = args.seed_offset
    obs = world.reset(episode, start=0)
    total = 0.0
    length = 0
    episodes = []
    agent.export(args.output / (args.method.lower() + "-initial.npz"))
    started = perf_counter()
    for step in range(1, args.steps + 1):
        action = agent.act(obs)
        next_obs, reward, end, trunc, info = world.step(
            **expand_action(action, args.method, args.speed_ceiling)
        )
        replay.add(obs, action, reward, next_obs, end)
        obs = next_obs
        total += reward
        length += 1
        if replay.count >= 256:
            agent.update(replay.sample())
        if end or trunc:
            episodes.append(
                dict(
                    episode=episode,
                    track=world.track.name,
                    steps=length,
                    return_value=total,
                    failed=bool(info["failed_vehicles"]),
                    collision=bool(info["collision_pairs"]),
                    track_exit=bool(info["outside_vehicles"]),
                )
            )
            episode += 1
            total = 0.0
            length = 0
            # Random track locations expose both straights and genuine corners.
            world = worlds[(episode - args.seed_offset) % len(worlds)]
            obs = world.reset(episode, start=float(rng.uniform(0, world.track.length)))
        if step % 1000 == 0 or step == args.steps:
            prefix = args.output / args.method.lower()
            agent.export(str(prefix) + ".npz")
            agent.export(str(prefix) + "-distribution.npz", distribution=True)
            torch.save(
                dict(
                    actor=agent.actor.state_dict(),
                    critics=agent.critics.state_dict(),
                    targets=agent.targets.state_dict(),
                    actor_optimizer=agent.actor_optimizer.state_dict(),
                    critic_optimizer=agent.critic_optimizer.state_dict(),
                    steps=previous_steps + step,
                    torch_rng=torch.get_rng_state(),
                    numpy_rng=rng.bit_generator.state,
                ),
                str(prefix) + "-training.pt",
            )
            manifest = dict(
                protocol=PROTOCOL,
                method=args.method,
                steps=previous_steps + step,
                gradient_updates=previous_updates + max(0, step - 255),
                stage_steps=step,
                speed_reference_range=[20, args.speed_ceiling] if args.method == "HILEPP" else None,
                pretraining={key: value for key, value in prior.items() if key != "episodes"}
                if prior
                else None,
                replay_reset_at_stage_start=bool(prior),
                tracks=args.tracks,
                architecture=ARCHITECTURE,
                encoder=dict(
                    context_features=11,
                    opponent_features=4,
                    phi=[4, 64, 32],
                    pooling="sum",
                    rho=[32, 32, 32],
                    concatenated_features=43,
                    actor_and_critics=True,
                ),
                vehicle_limits=[asdict(v) for v in world.limits],
                training_seed=args.seed,
                training_episode_seeds=[args.seed_offset, episode],
                observations=31,
                actions=actions,
                episode_duration_s=args.episode_duration,
                dt=world.config.dt,
                planner=asdict(world.config),
                reward="progress speed / 200 + number of opponents behind; termination penalty -10",
                entropy=0.03,
                discount=0.99,
                learning_rate=0.0003,
                hidden_layers=[64, 64],
                elapsed_s=(prior.get("elapsed_s", 0) if prior else 0) + perf_counter() - started,
                stage_elapsed_s=perf_counter() - started,
                episodes=episodes,
                distribution_sha256=hashlib.sha256(
                    Path(str(prefix) + "-distribution.npz").read_bytes()
                ).hexdigest(),
                checkpoint_sha256=hashlib.sha256(
                    Path(str(prefix) + ".npz").read_bytes()
                ).hexdigest(),
            )
            Path(str(prefix) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                args.method,
                previous_steps + step,
                "episodes",
                episode,
                "last",
                episodes[-1] if episodes else None,
                "seconds",
                round(perf_counter() - started),
                flush=True,
            )


if __name__ == "__main__":
    main()
