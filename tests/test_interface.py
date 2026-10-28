import numpy as np
import pytest
from hilepp.interface import HileppEnv

from hilepp.policy import limit_lateral_reference
from vehiclegym.planner.references import FrenetReference


def test_lateral_reference_limit_preserves_policy_direction_and_other_parameters():
    raw = FrenetReference(speed=31, lateral=-4, speed_weight=2, lateral_weight=3)
    limited = limit_lateral_reference(raw, measured_lateral=0.5)
    assert limited == FrenetReference(speed=31, lateral=-0.5, speed_weight=2, lateral_weight=3)
    assert raw.lateral == -4
    assert limit_lateral_reference(raw, measured_lateral=-4.2) == raw
    with pytest.raises(ValueError):
        limit_lateral_reference(raw, 0, max_offset=0)


def test_hilepp_i_maps_to_physical_references():
    env = HileppEnv("I")
    reference = env.expand_action([-1, 1])
    assert reference.speed == 18
    assert reference.lateral == 4.5
    assert reference.speed_weight == 0.5
    assert reference.lateral_weight == 1


def test_hilepp_ii_also_maps_cost_weights():
    reference = HileppEnv("II").expand_action([1, -1, -1, 1])
    assert (reference.speed, reference.lateral) == (32, -4.5)
    assert (reference.speed_weight, reference.lateral_weight) == (0.1, 10)


@pytest.mark.parametrize("action", [[0], [np.nan, 0], [2, 0]])
def test_invalid_actions_raise(action):
    with pytest.raises(ValueError):
        HileppEnv().expand_action(action)


def test_observation_and_truncation_contract():
    env = HileppEnv(max_steps=1)
    obs, info = env.reset(42, {"scenario": "blocking"})
    assert obs.shape == (22,) and obs.dtype == np.float32
    assert info["seed"] == 42
    next_obs, reward, terminated, truncated, info = env.step([0, 0])
    assert np.isfinite(next_obs).all() and np.isfinite(reward)
    assert truncated and not terminated


def test_wrapper_uses_policy_references(monkeypatch):
    from hilepp.policy import HierarchicalPolicy

    env = HileppEnv()
    env.reset(42, {"scenario": "blocking"})
    captured = []

    def record_plan(state, opponents, reference):
        captured.append(reference)
        return "planner result"

    monkeypatch.setattr(env.world.planner, "plan", record_plan)
    wrapper = HierarchicalPolicy(lambda observation: [0, 0.5], env)
    assert wrapper.plan() == "planner result"
    assert captured[0].speed == 25
    assert captured[0].lateral == 2.25
