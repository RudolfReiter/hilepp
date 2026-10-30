import numpy as np
import pytest
import torch
from hilepp.sac import SAC
from hilepp.policy import NumpyPolicy, NumpyGaussianPolicy


def reordered(observation):
    return torch.cat(
        (
            observation[..., :11],
            observation[..., 11:].reshape(-1, 5, 4)[:, [3, 0, 4, 1, 2]].flatten(-2),
        ),
        dim=-1,
    )


def test_actor_and_both_critics_are_invariant_and_encoders_learn():
    agent = SAC(observations=31, actions=4, seed=7, deep_sets=True)
    torch.nn.init.normal_(agent.actor.mean.weight, std=0.1)
    observation = torch.randn(12, 31)
    permuted = reordered(observation)
    torch.testing.assert_close(agent.actor(observation, True)[0], agent.actor(permuted, True)[0])
    actions = torch.randn(12, 4).tanh()
    for critic in agent.critics:
        torch.testing.assert_close(
            critic(torch.cat((observation, actions), -1)),
            critic(torch.cat((permuted, actions), -1)),
        )
    before = agent.actor.encoder.phi[0].weight.detach().clone()
    agent.update([observation, actions, torch.ones(12, 1), torch.randn(12, 31), torch.zeros(12, 1)])
    assert not torch.equal(before, agent.actor.encoder.phi[0].weight)
    for network in [agent.actor, *agent.critics]:
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in network.encoder.parameters()
        )


@pytest.mark.parametrize("opponents", [0, 1, 5, 8])
def test_numpy_export_preserves_set_encoding_and_sampling(tmp_path, opponents):
    agent = SAC(observations=31, actions=4, seed=3, deep_sets=True)
    torch.nn.init.normal_(agent.actor.mean.weight, std=0.2)
    observation = np.random.default_rng(42).normal(size=11 + 4 * opponents).astype(np.float32)
    mean, full = tmp_path / "mean.npz", tmp_path / "full.npz"
    agent.export(mean)
    agent.export(full, distribution=True)
    policy = NumpyPolicy(mean)
    assert policy.deep_sets
    np.testing.assert_allclose(policy(observation), agent.act(observation, True), atol=1e-6)
    permuted = np.r_[observation[:11], observation[11:].reshape(opponents, 4)[::-1].ravel()]
    np.testing.assert_allclose(policy(observation), policy(permuted), atol=1e-6)
    first, second = NumpyGaussianPolicy(full, 21), NumpyGaussianPolicy(full, 21)
    np.testing.assert_allclose(first(observation), second(permuted), atol=1e-6)
