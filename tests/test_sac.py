import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hilepp.sac import SAC, Replay  # noqa: E402 — optional training dependency
from hilepp.policy import NumpyPolicy  # noqa: E402


def test_sac_update_and_portable_export(tmp_path):
    agent = SAC(seed=3)
    replay = Replay(capacity=64)
    rng = np.random.default_rng(5)
    for i in range(64):
        replay.add(rng.normal(size=22), rng.uniform(-1, 1, 2), 1.0, rng.normal(size=22), i % 4 == 0)
    before = agent.actor.mean.weight.detach().clone()
    losses = agent.update(replay.sample(32))
    assert all(np.isfinite(value) for value in losses.values())
    assert not torch.equal(before, agent.actor.mean.weight)
    checkpoint = tmp_path / "actor.npz"
    agent.export(checkpoint)
    obs = rng.normal(size=(5, 22)).astype(np.float32)
    np.testing.assert_allclose(
        NumpyPolicy(checkpoint)(obs), agent.act(obs, deterministic=True), atol=2e-6
    )


def test_replay_preserves_terminal_mask():
    replay = Replay(capacity=2)
    replay.add(np.zeros(22), np.zeros(2), 1.0, np.ones(22), False)
    replay.add(np.zeros(22), np.zeros(2), 1.0, np.ones(22), True)
    np.testing.assert_array_equal(replay.terminated.ravel(), [0, 1])
