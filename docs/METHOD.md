# Method and deviations from ECC 2023

This repository implements the HILEPP-II interface: four SAC outputs set the speed and lateral references and their cost weights in a nonlinear MPC. The MPC is part of the environment during SAC training; no gradients are propagated through its solve. Direct SAC instead uses two physical actuator actions and bypasses the ego MPC entirely. The MPC baseline uses fixed references and weights.

## Shared race

Six vehicles use the original vehicle_gym kinematic Frenet model. All five opponents are independently optimized MPC policies. A measured-state snapshot is supplied to every controller before any state advances. Each controller solves at every 0.2 s step; only its first control is applied. The previous solution is only a numerical warm start.

The MPC has 25 shooting intervals and a 5 s horizon, acados SQP, soft state/acceleration/obstacle constraints and hard actuator limits. The approximate terminal condition limits speed to 15 m/s and heading error to 0.02 rad, with soft penalties. Multiple initial guesses may be tried for the same optimization problem. See the foundation's `RaceMPC` implementation for exact costs, penalties and numerical tolerances.

Obstacle assumptions follow the paper's structure: measured speed and heading error for cars ahead/alongside, full braking at 10 m/s² for cars more than 4.6 m behind. Integration uses RK4, clips lateral positions to the usable corridor and stops braking motion at standstill. The other MPCs do not actually adopt these assumed trajectories. The recreated implementation does not establish the paper's recursive feasibility guarantees.

## Training

Both actors are trained from scratch with twin-critic SAC, two hidden layers of 64 ReLU units, discount 0.99, learning rate 0.0003, fixed entropy coefficient 0.03, Polyak factor 0.005 and batch size 128. Updates start after 256 transitions. Training uses random initial circuit locations and 60 s episode limits. Both policies receive 12,000 preliminary interactions with seed 42, followed by 6,000 interactions with seed 43 and episode seeds starting at 5000. The full networks and optimizer states are transferred; the replay buffer is reset. HILEPP’s preliminary speed-reference range is 20–34 m/s. The final stage and evaluation use 20–28 m/s, matching the pure MPC baseline’s 28 m/s target and leaving two faster opponents at 31 and 29 m/s. Direct SAC’s actuator interface is unchanged across stages. Rewards combine progress speed divided by 200 and the number of opponents behind, with −10 at termination. There are no specialist blocking rewards or reference governors.

Checkpoints record their training interaction count, gradient updates, architecture, exact planner configuration and hash. Test seeds 20000–20002 are reserved from training. The published policies use the final training checkpoint; evaluation seeds are not used to choose checkpoints. Opponent controller parameters are fixed, but their actual actions respond to each method's evolving race.

## Actor execution and evaluation

SAC trains a squashed Gaussian policy. Published runs include both seeded samples from that full learned distribution and deterministic mean actions from the same final checkpoint. These are different control laws in a nonlinear environment. The full-policy files retain the learned log standard deviation and support NumPy inference; sampled runs use PCG64 with `policy_seed = 60000 + race_seed`. All means, dispersions and checkpoint hashes are retained. There is no manually chosen passing or blocking command.

Mean-action HILEPP completed IMS but became stuck behind traffic in this run. The sampled actor produced overtakes in validation and is the default visual replay. Both execution modes, all three fixed seeds and every failure remain selectable in the published results. These seeds were inspected during development; this is a diagnostic recreation, not an unseen generalization benchmark. The experiment does not establish that the paper’s learned defensive strategy has been recovered.

## Scope

IMS is the training and primary evaluation circuit, matching the circuit used by the original multi-agent example. Sepang supplies an additional tight-corner stress test. Its geometry and failure cases are retained rather than replaced by a synthetic gentle road.

This is a reduced-budget recreation, not the original repository or a quantitative replication. Differences include observation encoding, actuator-level direct SAC, 0.2 s/25-node discretization, numerical solver setup, soft constraint penalties, conservative constant road-width constraints, collision envelope and reward scaling. Collision detection uses oriented rectangular vehicle footprints. Any car's collision, track exit or rejected solve terminates the race.

The outcome table shows all test seeds. Duration, progress, final position and passes must be interpreted together; a short failed episode cannot be credited as a successful finish. Training and evaluation manifests are the authoritative record of measured results.
