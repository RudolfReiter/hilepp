"""Permutation-invariant opponent observations, concatenated with ordered ego data."""

import torch
from torch import nn

CONTEXT_FEATURES = 11  # Seven curvature samples, then four ego states.
OPPONENT_FEATURES = 4  # Relative progress, lateral position, speed, heading error.
SET_WIDTH = 32
ARCHITECTURE = "deep-sets-sum-v1"


class DeepSetEncoder(nn.Module):
    """[context, rho(sum_i phi(opponent_i))]; no opponent IDs or positional encoding.

    Input uses a flat transport buffer only. No learned layer ever receives the
    ordered concatenation of opponents. The set can have zero or more members.
    """

    output_features = CONTEXT_FEATURES + SET_WIDTH

    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(OPPONENT_FEATURES, 64), nn.ReLU(), nn.Linear(64, SET_WIDTH), nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(SET_WIDTH, SET_WIDTH), nn.ReLU(), nn.Linear(SET_WIDTH, SET_WIDTH), nn.ReLU()
        )

    def forward(self, observation):
        extra = observation.shape[-1] - CONTEXT_FEATURES
        if extra < 0 or extra % OPPONENT_FEATURES:
            raise ValueError("Expected 11 context features followed by four features per opponent")
        opponents = observation[..., CONTEXT_FEATURES:].reshape(
            *observation.shape[:-1], extra // OPPONENT_FEATURES, OPPONENT_FEATURES
        )
        pooled = self.phi(opponents).sum(dim=-2)
        return torch.cat((observation[..., :CONTEXT_FEATURES], self.rho(pooled)), dim=-1)


class SetCritic(nn.Module):
    """Each critic learns its own set encoder; actions enter after aggregation."""

    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.encoder = DeepSetEncoder()
        self.value = nn.Sequential(
            nn.Linear(self.encoder.output_features + actions, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state_action):
        features = self.encoder(state_action[..., : -self.actions])
        return self.value(torch.cat((features, state_action[..., -self.actions :]), dim=-1))
