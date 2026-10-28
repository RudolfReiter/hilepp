"""HILEPP-I/II action expansion and normalized observations (ECC 2023)."""

import numpy as np
from vehiclegym.planner.references import FrenetReference
from vehiclegym.simulator.racing_world import RacingWorld


class HileppEnv:
    """Gym-style API without requiring a specific RL framework."""

    observation_size = 22

    def __init__(self, variant="I", max_steps=80):
        if variant not in ("I", "II"):
            raise ValueError("variant must be I or II")
        self.variant = variant
        self.action_size = 2 if variant == "I" else 4
        self.world = RacingWorld(max_steps=max_steps)

    def reset(self, seed=0, options=None):
        options = options or {}
        obs = self.world.reset(seed, options.get("scenario", "overtaking"), options.get("bend"))
        return obs, {"seed": seed, "scenario": self.world.scenario}

    def expand_action(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,) or not np.isfinite(action).all():
            raise ValueError("Action has the wrong shape or contains non-finite values")
        if np.max(abs(action)) > 1 + 1e-6:
            raise ValueError("Normalized actions must be in [-1, 1]")
        action = np.clip(action, -1, 1)
        speed, lateral = 25 + 7 * action[0], 4.5 * action[1]
        wv, wn = (0.5, 1.0) if self.variant == "I" else (10 ** action[2], 10 ** action[3])
        return FrenetReference(float(speed), float(lateral), float(wv), float(wn))

    def step(self, action):
        return self.world.step(self.expand_action(action))


def gymnasium_env(variant="I", max_steps=80):
    """Optional real Gymnasium adapter for Stable-Baselines3 and other clients."""
    import gymnasium as gym

    class Adapter(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.env = HileppEnv(variant, max_steps)
            self.action_space = gym.spaces.Box(-1, 1, (self.env.action_size,), np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (22,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            rollout_seed = int(self.np_random.integers(0, 2**31 - 1))
            return self.env.reset(rollout_seed, options)

        def step(self, action):
            return self.env.step(action)

    return Adapter()
