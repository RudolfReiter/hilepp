"""ECC-style HILEPP-II and direct SAC interfaces to the same six-car race."""

import numpy as np
from vehiclegym.planner.references import FrenetReference


def expand_action(action, method, speed_ceiling=28.0):
    action = np.clip(np.asarray(action, dtype=float), -1, 1)
    if method == "HILEPP":
        if action.shape != (4,):
            raise ValueError(
                "HILEPP-II needs speed, lateral reference, speed weight, lateral weight"
            )
        return dict(
            reference=FrenetReference(
                speed=float((20 + speed_ceiling) / 2 + (speed_ceiling - 20) / 2 * action[0]),
                lateral=float(3.5 * action[1]),
                speed_weight=float(10 ** action[2]),
                lateral_weight=float(2 * 10 ** action[3]),
            )
        )
    if method == "SAC":
        if action.shape != (2,):
            raise ValueError("Direct SAC needs force and steering rate")
        return dict(direct_control=np.array([10000 * action[0] - 2000, 0.3 * action[1]]))
    if method == "MPC":
        return {}
    raise ValueError(method)
