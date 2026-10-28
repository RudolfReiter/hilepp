# HILEPP · recreated hierarchical racing policy

**A learned race strategy wrapped around vehicle_gym’s nonlinear motion planner.**

[Interactive research page](https://rudolfreiter.github.io/hilepp/) · [ECC 2023 paper](https://ieeexplore.ieee.org/document/10178143) · [Author manuscript](https://publications.syscop.de/Reiter2023.pdf) · [vehicle_gym](https://github.com/RudolfReiter/vehicle_gym)

![Recorded learned policy driving scene](docs/assets/driving-scene.png)

> **This is a recreated implementation, not the original HILEPP training repository.** The included checkpoint was trained afresh. It is not an original paper checkpoint, and these results do not reproduce the paper’s full benchmark.

## Research

**A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing**<br>
Rudolf Reiter, Jasper Hoffmann, Joschka Boedecker, Moritz Diehl.<br>
**European Control Conference (ECC 2023)**, Bucharest, pp. 1–8.<br>
DOI: [10.23919/ECC57647.2023.10178143](https://doi.org/10.23919/ECC57647.2023.10178143).

The policy chooses speed and lateral references. A reference generator expands them into a trajectory/cost specification, and nonlinear model predictive control computes the force and steering-rate commands. The policy does not directly output throttle or steering.

```text
Curvature + ego state + opponents
                  │ 22 observations
                  ▼
           SAC neural policy
                  │ HILEPP-I: v_ref, n_ref
                  │ HILEPP-II: also w_v, w_n
                  ▼
          Reference expansion
                  ▼
    vehicle_gym nonlinear motion planner
                  │ force, steering rate
                  ▼
          Original vehicle dynamics
```

## Install and replay

Use the **improved vehicle_gym 1.1** checkout beside this repository. The pinned foundation commit is recorded in `PROVENANCE.json`; `requirements-foundation.txt` provides a Git installation once that commit is pushed.

```bash
# Expected siblings: vehicle_gym/ and hilepp/
cd hilepp
python3 -m venv .venv
source .venv/bin/activate
pip install -e ../vehicle_gym -e '.[train,dev]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m hilepp.evaluate
python -m pytest
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. The committed page already includes replay data; Python is only needed to regenerate it. GitHub Pages must be enabled before the public page link becomes live.

For inference without PyTorch, install `pip install -e ../vehicle_gym -e .` and use the NumPy checkpoint:

```python
from hilepp.interface import HileppEnv
from hilepp.policy import NumpyPolicy, HierarchicalPolicy

env = HileppEnv("I", max_steps=64)
obs, _ = env.reset(seed=10000, options={"scenario": "overtaking"})
policy = NumpyPolicy("checkpoints/best.npz")
while not env.world.finished:
    obs, reward, terminated, truncated, info = env.step(policy(obs))

# Alternatively, request a plan without advancing the simulation:
env.reset(seed=10000)
plan = HierarchicalPolicy(policy, env).plan()
```

Any callable returning normalized actions can be supplied to `HierarchicalPolicy`. An optional `gymnasium_env()` adapter is available with `pip install -e '.[gym]'`; the framework-independent API is used in the included training and evaluation. The optional Gymnasium adapter has not been exercised locally.

## Train and evaluate

```bash
# Preserve the included checkpoint; train into a separate directory.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m hilepp.train \
  --steps 4000 --seed 7 --variant I --output checkpoints/new-run
python -m hilepp.evaluate --checkpoint-dir checkpoints/new-run --output artifacts/new-run

# HILEPP-II interface is implemented, but no trained II checkpoint is included.
python -m hilepp.train --variant II --steps 4000 --output checkpoints/variant-ii
```

SAC uses a squashed Gaussian actor, twin Q networks, target critics, fixed entropy coefficient 0.1, learning rate 3e-4, discount 0.99, Polyak coefficient 0.005, two hidden layers of width 64, batch size 128, and 256 random warmup steps. No differentiation through the planner is used. Time limits bootstrap; terminal failures do not. Checkpoints contain only the deterministic actor’s mean parameters, not optimizer/replay state; they are inference exports, not resumable training snapshots.

Training samples episode seeds 0–6999. Validation seeds 8000–8002 select checkpoints every 1000 interactions. Held-out seeds 10000–10004 are evaluated only after selection, on each of three tasks. The page always shows test seed 10000 for each method, not a handpicked successful run.

## Recorded results

4,000 interactions, one training seed (7), 16-second episodes, HILEPP-I. The selected validation checkpoint is step 4,000.

| Method | Mean return, all 15 test episodes | Mean final rank | Completed | Solver failures | Detected collisions |
|---|---:|---:|---:|---:|---:|
| Fixed-reference MPP | 162.19 | 1.20 | 15/15 | 0 | 0 |
| Initial neural policy | 151.28 | 1.33 | 15/15 | 0 | 0 |
| Trained SAC policy | 171.72 | 1.00 | 15/15 | 0 | 0 |

The trained policy improves mean return by **5.9% over fixed MPP** in this small paired evaluation. There are 45 test episodes total. Fixed MPP requests 30 m/s at track center; the learned policy can request 18–32 m/s and ±4.5 m, so the gain includes both speed and lateral-reference choices. These results establish a working basic implementation, not statistical recovery of the original experiment or a safety claim.

![Actual validation and held-out results](docs/assets/evaluation.svg)

Raw [training log](checkpoints/training.json), [all test records](docs/data/evaluation.json), and [recorded trajectories/plans](docs/data/experiment.json) are committed. Every scene, trace, and table derives from those files. Figure regeneration: `python scripts/build_report.py`.

## Fidelity and limitations

See [paper-to-code mapping](docs/METHOD.md) for equations and implementation details.

- Uses the original vehicle_gym five-state Frenet model and the paper’s separation between RL and motion planning.
- Implements both HILEPP-I and II action interfaces; only I is trained here.
- Recreates observation structure and progress/rank reward. A −10 termination penalty is added for solver failure, collision, or leaving the track.
- Replaces acados/HPIPM with CasADi/IPOPT, uses a three-second horizon, and omits the recursive-feasibility terminal set.
- Uses constant-speed/lateral opponent predictions and simple lane-tracking/car-following opponents, instead of the paper’s leader/follower prediction rules and MPC opponents.
- Uses Frenet obstacle ellipses with bounded slack and approximate Frenet body collision checks; it does not establish continuous Cartesian collision freedom.
- Uses synthetic circuit sections, one training seed, 4,000 interactions, and 16-second episodes. The paper used a substantially larger training/evaluation protocol.
- The failed-plan rule stops before executing the rejected control; it is not a fallback vehicle controller.

## Layout and validation

`src/hilepp/`: action interface, policy wrapper, NumPy inference, SAC, training, evaluation.<br>
`checkpoints/`: new initial, selected, and final policies plus training log.<br>
`tests/`: interface, numerical update, checkpoint parity, and wrapper contract tests.<br>
`docs/`: static research page, plots, and replay data.<br>
`scripts/`: data checks, interaction checks, actual-renderer export, and source archive build.

```bash
python -m pytest
python scripts/check_data.py
node scripts/check_interactions.mjs
node scripts/render_scene.mjs
# Optional PNG export requires system GI / Rsvg / Cairo / Pillow:
/usr/bin/python3 scripts/rasterize_scenes.py
python scripts/build_site.py
```

The visualization starts paused, interpolates recorded states, and holds each forecast for three seconds so the ghost and actual car can visibly diverge. The forecast comparison measures replanning consistency, not an independent learned predictor. The export renders the page’s Canvas commands into SVG; it is not a browser layout test.

MIT license. Citation metadata in [CITATION.cff](CITATION.cff). Dependency/source hashes and environment versions in [PROVENANCE.json](PROVENANCE.json).
