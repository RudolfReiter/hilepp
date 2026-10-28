# Paper-to-code mapping

Source: Reiter et al., **A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing**, ECC 2023. [Author PDF](https://publications.syscop.de/Reiter2023.pdf).

| Paper component | Implementation | Scope |
|---|---|---|
| Five-state Frenet dynamics | vehicle_gym `vehicle_models/model_kinematic_frenet.py` | Original model reused |
| High-level policy → reference generator → MPP | `interface.py`, `policy.py`; vehicle_gym `planner/references.py`, `planner/portable.py` | Same hierarchy |
| HILEPP-I reference actions | `HileppEnv.expand_action` | Speed 18–32 m/s; lateral −4.5–4.5 m |
| HILEPP-II reference and cost actions | same function, variant II | Adds positive speed/lateral weights 0.1–10 |
| Observation, Eq. (20) | vehicle_gym `RacingWorld.observation` | 7 curvature samples + 3 ego values + 3 × 4 opponent values |
| Reward, Eq. (21) | `RacingWorld.step` | Progress rate / 200 + opponents behind; added −10 terminal penalty |
| SAC training | `sac.py`, `train.py` | New implementation and newly trained weights |
| Overtaking / blocking / mixed tasks | `RacingWorld.reset` | Seeded simplified traffic scenarios |
| Multiple-seed evaluation | `evaluate.py` | Five paired test seeds per task, one training seed |

Normalized policy actions lie in `[-1,1]`. Variant I maps to `v_ref = 25 + 7a_0`, `n_ref = 4.5a_1`; fixed weights are 0.5 for speed and 1 for lateral position. Variant II maps `w_v = 10**a_2`, `w_n = 10**a_3`. These ranges are recreation choices, not claimed verbatim original training hyperparameters.

Observation curvature is sampled at distances `[0,10,20,40,60,90,120]` m and divided by 0.04. Ego `[n,v,alpha]` is divided by `[7,32,0.6]`. Opponent `[relative_s,n,v,alpha]` is divided by `[80,7,32,0.6]`; opponent ordering is fixed within the episode, not sorted each step.

The reference expansion returns original `ReferenceValues` matrices: `Q = diag(0.015,w_n,2,w_v,2)` and `R = diag(1e-8,0.4)` in physical force units. The portable optimizer applies the corresponding quadratic tracking penalties, with force scaled to kN internally. The longitudinal reference advances at the requested speed. The final state receives tracking cost but no terminal invariant-set constraint.

The paper’s recursive-feasibility terminal construction and leader/follower obstacle prediction logic are omitted. The portable implementation uses Frenet ellipses with longitudinal/lateral radii 7 m / 3 m, one obstacle slack per stage bounded at 0.08, and penalty `1e4 sigma + 1e4 sigma^2`. Reported constraint residual includes slack, so a small residual alone does not mean zero geometric relaxation.

The original paper used many more interactions and training seeds. This included run is 4,000 interactions with seed 7, and a 64-step episode limit. Best-checkpoint selection uses separate validation seeds. The test set is evaluated only after selection. Reproduction should compare deterministic trajectories, rewards, and solver status within numerical tolerances; wall-clock solve times depend on hardware.

SAC follows the standard off-policy, entropy-regularized formulation with twin critics and target critics. [Primary reference: Haarnoja et al., 2018](https://proceedings.mlr.press/v80/haarnoja18b.html). The compact implementation uses a fixed entropy coefficient rather than automatic temperature tuning.
