# HILEPP — hierarchical learning-based predictive racing

Recreated implementation of **A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing**, Rudolf Reiter, Jasper Hoffmann, Joschka Boedecker and Moritz Diehl, **ECC 2023**.

**This is not the original HILEPP repository.** The controllers, training runs and recorded experiments here are a new recreation, with a smaller training budget and a documented numerical implementation.

[Interactive comparison](https://rudolfreiter.github.io/hilepp/) · [Paper](https://publications.syscop.de/Reiter2023.pdf) · [vehicle_gym foundation](https://github.com/RudolfReiter/vehicle_gym)

![Recorded six-car race](docs/assets/driving-scene.png)

## What is compared

| Policy | Ego controller | Five opponents |
|---|---|---|
| **HILEPP** | HILEPP-II: SAC sets speed/lateral references and their weights; nonlinear MPC computes actuator commands | Independent racing MPCs |
| **SAC** | SAC directly sets longitudinal force and steering rate; no ego MPC | Independent racing MPCs |
| **MPC** | The same nonlinear MPC with fixed references and weights; no RL | Independent racing MPCs |

Every MPC replans at every 0.2 s control step using the current measured states. All six vehicles then advance simultaneously. Opponents are never scripted lane followers, and no policy has access to their future controls.

The viewer shows actual circuit geometry, measured race order and speed, the selected controller's current horizon, and its assumptions about the five other cars. Once an opponent is fully behind, its assumed trajectory changes from constant speed to full braking. This models its ability to yield, not its intended action. There are no held forecast comparisons or ghost cars.

## Install

Create sibling `vehicle_gym` and `hilepp` checkouts, with the foundation revision pinned in `requirements-foundation.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ../vehicle_gym -e '.[train,dev]'
bash ../vehicle_gym/scripts/setup_acados.sh
# Export the ACADOS_SOURCE_DIR and LD_LIBRARY_PATH printed by the script.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

## Train and evaluate

```bash
bash scripts/train_curriculum.sh
python -m hilepp.evaluate_circuit --tracks IMS Sepang --duration 180 \
  --seeds 20000 20001 20002 --output docs/data
python scripts/check_data.py
python scripts/build_report.py
python -m http.server 8000 --directory docs
```

The included `.npz` actors support NumPy inference without PyTorch or pickle. Files ending in `-distribution.npz` retain SAC’s learned mean and log standard deviation. The viewer defaults to seeded samples from this full policy; its **Actor action** selector also exposes deterministic mean-action results from the same checkpoint. MPC is unchanged between these views. Training uses the small SAC implementation in `src/hilepp/sac.py`; its critics, target critics and optimizer states are also saved locally. Published actor files and their manifests have SHA-256 checksums. Training does not use scripted maneuver rewards, imitation labels, pretrained paper weights, or an actuator fallback.

Training uses random starting locations on IMS and 60 s episodes. Both policies receive 12,000 preliminary interactions and 6,000 further interactions. HILEPP initially uses a 20–34 m/s reference range; its final stage restricts this to 20–28 m/s, matching the fixed MPC target of 28 m/s. The faster MPC opponents target 31 and 29 m/s. SAC uses direct actuator actions throughout. The stage transition preserves actor, critics and optimizer states, resets the replay buffer, and uses a separate seed range. Evaluation uses three separately reserved seeds, 180 s limits, and matched initial conditions. Both sampled-policy and mean-action execution are published; the fixed evaluation seeds were inspected during development and are not an unseen generalization benchmark. Sepang is an additional circuit stress test, outside the training distribution. Every run is retained, including collisions, track exits and solver failures by opponents. Final position alone is not meaningful when survival times differ. The experiment does not establish that HILEPP outperforms MPC or reproduce the paper's reported results.

## Recorded results

Sampled HILEPP completes all three 180-second IMS runs and overtakes two vehicles in each. The same checkpoint’s mean actions complete the races but become stuck behind traffic. Pure MPC completes all three IMS races; direct SAC exits the track. Sepang remains a stress test with failures. Both actor modes and every fixed seed are retained in the viewer, rather than presenting only the favorable execution mode.

These are diagnostic results from a small recreation. They do not establish recovery of the paper’s learned blocking strategy.

## HILEPP-II interface

The observation contains 31 normalized values: seven upcoming curvature samples; ego lateral position, speed, heading error and steering angle; and relative longitudinal position, lateral position, speed and heading error for each opponent.

For normalized actions `a ∈ [−1, 1]⁴`:

```text
speed reference    = 24 + 4 a₀       m/s
lateral reference  = 3.5 a₁          m
speed cost weight  = 10^a₂
lateral cost weight = 2 × 10^a₃
```

Direct SAC has two actions: `force = 10000 a₀ − 2000 N`, `steering rate = 0.3 a₁ rad/s`. Its output goes directly to the same vehicle dynamics. The pure-MPC ego uses 28 m/s, zero lateral offset, and weights 1 and 2.

Reward is longitudinal progress speed divided by 200, plus the number of opponents behind; termination adds −10. A shared snapshot, dynamics, opponent profiles and physical checks keep the policy comparisons consistent. Older short-road modules remain available through the explicitly named `hilepp-train-legacy` and `hilepp-evaluate-legacy` commands and are not used here. The default console commands now run the six-car circuit experiment.

See [method and deviations](docs/METHOD.md) and [control/prediction semantics](docs/CONTROL_AND_PREDICTION.md).

## Verify

```bash
python -m pytest
ruff check src tests scripts
python scripts/check_data.py
node scripts/check_interactions.mjs
node scripts/render_scene.mjs
python scripts/build_site.py
```

Committed data can be checked without acados. Regenerating races requires its compiled backend. JavaScript interaction tests use DOM/Canvas doubles; they are not browser layout tests.

## Citation

Reiter, R., Hoffmann, J., Boedecker, J., and Diehl, M. **A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing.** European Control Conference, 2023. [DOI: 10.23919/ECC57647.2023.10178143](https://doi.org/10.23919/ECC57647.2023.10178143).
