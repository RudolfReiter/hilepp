from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("torch")
from hilepp.train_defense import defense_bonus  # noqa: E402


def test_auxiliary_bonus_tracks_lane_coverage_without_changing_the_environment():
    world = SimpleNamespace(
        state=np.array([0, 0, 0, 32, 0.0]),
        opponents=np.array([[-15, 0, 0, 33, 0.0], [-30, 4, 0, 35, 0.0]]),
    )
    before = world.opponents.copy()
    assert defense_bonus(world, {"held_up": [True, False]}) == pytest.approx(1.5)
    np.testing.assert_array_equal(world.opponents, before)
    world.state[1] = -4.2
    assert defense_bonus(world, {"held_up": [False, False]}) == pytest.approx(0)
    world.opponents[:, 0] = [20, 40]
    assert defense_bonus(world, {"held_up": [False, False]}) == pytest.approx(0)
