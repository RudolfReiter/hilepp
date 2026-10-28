"""Inference and deployment wrapper, independent of the training framework."""

from pathlib import Path
import numpy as np


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
        return env.world.planner.plan(env.world.state, env.world.opponents[:, [0, 1, 3]], reference)
