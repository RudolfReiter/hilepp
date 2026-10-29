import numpy as np
import pytest
from hilepp.circuit import expand_action


def test_hilepp_ii_outputs_four_mpc_parameters():
    reference = expand_action(np.array([1, -1, 0.5, -0.5]), "HILEPP")["reference"]
    assert reference.speed == 28
    assert reference.lateral == -3.5
    np.testing.assert_allclose(reference.speed_weight, 10**0.5)
    np.testing.assert_allclose(reference.lateral_weight, 2 * 10**-0.5)


def test_direct_sac_has_no_reference_or_planner_output():
    output = expand_action([-1, 1], "SAC")
    assert set(output) == {"direct_control"}
    np.testing.assert_allclose(output["direct_control"], [-12000, 0.3])
    np.testing.assert_allclose(expand_action([1, -1], "SAC")["direct_control"], [8000, -0.3])


def test_policy_shapes_are_not_interchangeable():
    with pytest.raises(ValueError):
        expand_action([0, 0], "HILEPP")
    with pytest.raises(ValueError):
        expand_action([0, 0, 0, 0], "SAC")


def test_full_gaussian_actor_preserves_seeded_sampling(tmp_path):
    from hilepp.sac import SAC
    from hilepp.policy import NumpyGaussianPolicy, NumpyPolicy

    agent = SAC(observations=31, actions=4, seed=21)
    full = tmp_path / "distribution.npz"
    mean = tmp_path / "mean.npz"
    agent.export(full, distribution=True)
    agent.export(mean)
    observation = np.linspace(-0.5, 0.5, 31).astype(np.float32)
    first = NumpyGaussianPolicy(full, seed=71)
    second = NumpyGaussianPolicy(full, seed=71)
    a = np.array([first(observation) for _ in range(8)])
    b = np.array([second(observation) for _ in range(8)])
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all() and np.max(abs(a)) <= 1
    assert np.max(np.ptp(a, axis=0)) > 0.05
    np.testing.assert_allclose(
        NumpyPolicy(mean)(observation), agent.act(observation, deterministic=True), atol=1e-6
    )
