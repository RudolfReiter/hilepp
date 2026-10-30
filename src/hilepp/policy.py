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


def _load_actor(checkpoint, distribution=False):
    with np.load(Path(checkpoint), allow_pickle=False) as data:
        parameters = {key: data[key].copy() for key in data.files}
    layers = ["body.0", "body.2", "mean"] + (["log_std"] if distribution else [])
    deep_sets = any(key.startswith("encoder.") for key in parameters)
    if deep_sets:
        layers += ["encoder.phi.0", "encoder.phi.2", "encoder.rho.0", "encoder.rho.2"]
    expected = {f"{layer}.{kind}" for layer in layers for kind in ["weight", "bias"]}
    if set(parameters) != expected or not all(np.isfinite(v).all() for v in parameters.values()):
        raise ValueError("Invalid policy checkpoint")
    return parameters, deep_sets


def _hidden(parameters, observation, deep_sets):
    def layer(value, prefix):
        return np.maximum(
            value @ parameters[prefix + ".weight"].T + parameters[prefix + ".bias"], 0
        )

    value = np.asarray(observation, dtype=np.float32)
    if deep_sets:
        extra = value.shape[-1] - 11
        if extra < 0 or extra % 4:
            raise ValueError("Expected 11 context features and four features per opponent")
        opponents = value[..., 11:].reshape(*value.shape[:-1], extra // 4, 4)
        pooled = layer(layer(opponents, "encoder.phi.0"), "encoder.phi.2").sum(axis=-2)
        encoded = layer(layer(pooled, "encoder.rho.0"), "encoder.rho.2")
        value = np.concatenate((value[..., :11], encoded), axis=-1)
    return layer(layer(value, "body.0"), "body.2")


class NumpyPolicy:
    """Mean actor inference, including the learned Deep Sets encoder when present."""

    def __init__(self, checkpoint):
        self.parameters, self.deep_sets = _load_actor(checkpoint)

    def __call__(self, observation):
        value = _hidden(self.parameters, observation, self.deep_sets)
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
        self.parameters, self.deep_sets = _load_actor(checkpoint, distribution=True)
        self.reset_rng(seed)

    def reset_rng(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation):
        value = _hidden(self.parameters, observation, self.deep_sets)
        mean = value @ self.parameters["mean.weight"].T + self.parameters["mean.bias"]
        log_std = np.clip(
            value @ self.parameters["log_std.weight"].T + self.parameters["log_std.bias"], -5, 1
        )
        return np.tanh(mean + np.exp(log_std) * self.rng.standard_normal(mean.shape))
