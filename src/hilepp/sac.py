"""Small SAC implementation: squashed Gaussian actor, twin critics, target critics.

The planner is treated as part of the environment. No gradients are propagated
through the nonlinear solver. Time-limit truncations bootstrap; terminations do not.
"""

from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(inputs, outputs, width=64):
    return nn.Sequential(
        nn.Linear(inputs, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, outputs),
    )


class Actor(nn.Module):
    def __init__(self, observations=22, actions=2, width=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observations, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.mean = nn.Linear(width, actions)
        self.log_std = nn.Linear(width, actions)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        self.mean.bias.data[0] = 0.65
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -1.0)

    def forward(self, observations, deterministic=False):
        hidden = self.body(observations)
        mean = self.mean(hidden)
        if deterministic:
            return mean.tanh(), None
        distribution = Normal(mean, self.log_std(hidden).clamp(-5, 1).exp())
        sample = distribution.rsample()
        action = sample.tanh()
        log_prob = (distribution.log_prob(sample) - torch.log(1 - action.square() + 1e-6)).sum(
            -1, keepdim=True
        )
        return action, log_prob


class Replay:
    def __init__(self, observations=22, actions=2, capacity=50000, seed=0):
        self.obs = np.zeros((capacity, observations), np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((capacity, actions), np.float32)
        self.rewards = np.zeros((capacity, 1), np.float32)
        self.terminated = np.zeros((capacity, 1), np.float32)
        self.count, self.capacity = 0, capacity
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.count % self.capacity
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        self.count += 1

    def sample(self, batch=128):
        indices = self.rng.integers(0, min(self.count, self.capacity), batch)
        return [
            torch.from_numpy(array[indices])
            for array in (self.obs, self.actions, self.rewards, self.next_obs, self.terminated)
        ]


class SAC:
    def __init__(
        self,
        observations=22,
        actions=2,
        seed=0,
        learning_rate=3e-4,
        discount=0.99,
        polyak=0.005,
        entropy=0.1,
    ):
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.actor = Actor(observations, actions)
        self.critics = nn.ModuleList([mlp(observations + actions, 1) for _ in range(2)])
        self.targets = deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=learning_rate)
        self.discount, self.polyak, self.entropy = discount, polyak, entropy

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            return self.actor(torch.as_tensor(obs, dtype=torch.float32), deterministic)[0].numpy()

    def update(self, batch):
        obs, actions, rewards, next_obs, terminated = batch
        with torch.no_grad():
            next_actions, log_prob = self.actor(next_obs)
            next_inputs = torch.cat((next_obs, next_actions), dim=-1)
            next_q = torch.minimum(*[q(next_inputs) for q in self.targets])
            target = rewards + self.discount * (1 - terminated) * (next_q - self.entropy * log_prob)
        inputs = torch.cat((obs, actions), dim=-1)
        critic_loss = sum((q(inputs) - target).square().mean() for q in self.critics)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critics.requires_grad_(False)
        policy_actions, log_prob = self.actor(obs)
        policy_q = torch.minimum(
            *[q(torch.cat((obs, policy_actions), dim=-1)) for q in self.critics]
        )
        actor_loss = (self.entropy * log_prob - policy_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critics.requires_grad_(True)
        with torch.no_grad():
            for target_network, network in zip(self.targets, self.critics):
                for target_parameter, parameter in zip(
                    target_network.parameters(), network.parameters()
                ):
                    target_parameter.lerp_(parameter, self.polyak)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
        }

    def export(self, destination, distribution=False):
        """Safe NumPy inference; optionally retain the learned Gaussian dispersion."""
        parameters = {
            name: tensor.detach().numpy()
            for name, tensor in self.actor.state_dict().items()
            if distribution or not name.startswith("log_std")
        }
        np.savez(destination, **parameters)
