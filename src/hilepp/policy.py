"""Inference and deployment wrapper, independent of the training framework."""

from pathlib import Path
from dataclasses import replace
import numpy as np


def limit_lateral_reference(reference, measured_lateral, max_offset=1.0):
    """Limit the requested lateral offset from the measured position.

    This optional reference generator moderates abrupt strategic commands. It
    neither replaces the neural policy nor establishes recursive feasibility.
    """
    if not np.isfinite(max_offset) or max_offset <= 0 or not np.isfinite(measured_lateral):
        raise ValueError("Reference limit must be positive and the measurement finite")
    return replace(
        reference,
        lateral=float(
            np.clip(reference.lateral, measured_lateral - max_offset, measured_lateral + max_offset)
        ),
    )


class NumpyPolicy:
    def __init__(self, checkpoint):
        with np.load(Path(checkpoint), allow_pickle=False) as data:
            self.parameters = {key: data[key].copy() for key in data.files}
        expected = {
            "body.0.weight",
            "body.0.bias",
            "body.2.weight",
            "body.2.bias",
            "mean.weight",
            "mean.bias",
        }
        if set(self.parameters) != expected or not all(
            np.isfinite(v).all() for v in self.parameters.values()
        ):
            raise ValueError("Invalid policy checkpoint")

    def __call__(self, observation):
        value = np.asarray(observation)
        for layer in ["body.0", "body.2"]:
            value = np.maximum(
                value @ self.parameters[layer + ".weight"].T + self.parameters[layer + ".bias"], 0
            )
        return np.tanh(value @ self.parameters["mean.weight"].T + self.parameters["mean.bias"])


class HierarchicalPolicy:
    """Use any callable normalized policy to parameterize vehicle_gym's planner."""

    def __init__(self, policy, environment):
        self.policy, self.environment = policy, environment

    def plan(self):
        env = self.environment
        action = np.asarray(self.policy(env.world.observation()))
        reference = env.expand_action(action)
        return env.world.planner.plan(env.world.state, env.world.opponents, reference)


class NumpyGaussianPolicy:
    """The full squashed-Gaussian SAC actor, with an explicit reproducible RNG.

    A mean-action replay and a sampled-policy replay are different execution
    modes of the same checkpoint. Neither inserts a maneuver or actuator filter.
    """

    def __init__(self, checkpoint, seed=0):
        with np.load(Path(checkpoint), allow_pickle=False) as data:
            self.parameters = {key: data[key].copy() for key in data.files}
        expected = {
            f"{layer}.{kind}"
            for layer in ["body.0", "body.2", "mean", "log_std"]
            for kind in ["weight", "bias"]
        }
        if set(self.parameters) != expected or not all(
            np.isfinite(value).all() for value in self.parameters.values()
        ):
            raise ValueError("Invalid Gaussian actor checkpoint")
        self.reset_rng(seed)

    def reset_rng(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation):
        value = np.asarray(observation)
        for layer in ["body.0", "body.2"]:
            value = np.maximum(
                value @ self.parameters[layer + ".weight"].T + self.parameters[layer + ".bias"], 0
            )
        mean = value @ self.parameters["mean.weight"].T + self.parameters["mean.bias"]
        log_std = np.clip(
            value @ self.parameters["log_std.weight"].T + self.parameters["log_std.bias"], -5, 1
        )
        return np.tanh(mean + np.exp(log_std) * self.rng.standard_normal(mean.shape))
