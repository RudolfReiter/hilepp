# Method and deviations from ECC 2023

HILEPP-II uses four SAC outputs to set speed/lateral references and their cost weights in a nonlinear MPC. The planner is part of the environment during training; gradients do not pass through its solve. Direct SAC uses two actuator actions and bypasses the ego MPC. The MPC baseline uses fixed references and weights.

## Permutation-invariant observations

The transport buffer contains seven normalized curvature previews, four ego states and four measured features for each of five opponents. It is split into ordered context and an unordered opponent set before reaching a learned layer. Shared `phi: 4 → 64 → 32`, sum pooling and `rho: 32 → 32 → 32` produce a 32-dimensional opponent embedding. Concatenation with the 11 context values produces 43 actor features. Each twin critic has its own learned encoder and concatenates actions after aggregation. All hidden layers use ReLU; the actor and value heads use two 64-unit hidden layers.

Permuting opponents changes neither action distributions nor Q values, within numerical tolerance. There are no opponent IDs or position-dependent embeddings. The safe NumPy exports include both Deep Sets mappings and reproduce the PyTorch mean policy. Seeded sampled inference additionally retains the learned log standard deviation. Tests verify invariance, nonzero encoder learning gradients, varying set sizes and export parity. Variable opponent count is supported by the encoder; published training and evaluation still use five opponents.

This implements the requested Deep Sets observation encoder with newly trained weights. It does not claim to recover the original implementation’s checkpoints.

## Shared periodic race

The six cars use the original vehicle_gym kinematic Frenet equations and common mass/drag parameters, with different per-car controller envelopes. Each MPC solves from a shared snapshot every 0.2 s, applies its first force and steering-rate command, and advances simultaneously with the others. Warm starts are numerical initial guesses only.

Every optimization has 25 shooting intervals and a five-second horizon, acados SQP, soft state/lateral-acceleration/obstacle constraints, hard force/steering-rate bounds and hard longitudinal acceleration constraints. The latter use net acceleration after drag and rolling resistance, including the first control interval. Positive minimum speeds are soft and can relax when yielding. The terminal condition softly limits speed to 15 m/s and heading error to 0.02 rad. Several initial guesses may be tried for the same problem.

Track curvature is evaluated modulo circuit length at each RK stage; progress and race position remain unwrapped. This avoids running beyond a finite track extension after several laps. Geometric collision checks use wrapped Cartesian positions. Pass detection has a 4.6 m clearance and distinguishes lap traffic from ordinary order changes. Opponent positions use the nearest lap relative to each observing car.

A leading/alongside opponent retains measured speed and heading error. A follower more than 4.6 m behind is assumed to brake at its own maximum deceleration until standstill. This is a yielding feasibility assumption, not the other MPC’s intended control. It does not establish recursive safety.

## Training and execution

Both actors train from scratch using twin-critic SAC: discount 0.99, learning rate 0.0003, entropy coefficient 0.03, Polyak factor 0.005 and batch size 128. Updates begin after 256 transitions. Training alternates episodes between Oval and IMS, randomizes starting locations and uses a 90 s episode limit. Each actor receives 12,000 interactions with seed 52 and environment episode seeds beginning at 1000. No earlier flat-MLP weights, imitation labels, blocking rewards or reference governors are used.

HILEPP uses a 20–28 m/s speed-reference range, matching the MPC ego’s 28 m/s target. Reward is longitudinal progress speed divided by 200 plus the number of opponents behind; physical termination adds −10. Time-limit truncation bootstraps. Every training checkpoint records architecture, vehicle limits, training seeds, step/update counts and SHA-256 hashes.

Both seeded samples and deterministic mean actions from the final checkpoint are evaluated. Sampled policies use NumPy PCG64 with `policy_seed = 60000 + race_seed`. These are two execution modes of the same learned actor, not additional policy algorithms. The pure MPC baseline is reused between modes after checking identical protocol, limits, starts, profiles and duration. Final checkpoints are used without choosing a best evaluation seed or checkpoint.

## Scope

The oval is an explicitly synthetic 125 m × 80 m semi-axis ellipse, about 650 m in circumference. Its 240 s horizon is long enough for multiple laps and repeated encounters. IMS and Sepang reuse the packaged vehicle_gym circuit datasets, smoothed over 6 m, with 180 s race limits. Sepang is outside the training circuit set.

All evaluation seeds 30000–30002 and every failed run are retained. These fixed seeds may be inspected during development and are not an unseen generalization benchmark. Any vehicle’s physical collision, track exit or rejected solve ends the race. A high finishing position in a short failed run is not a successful race finish. The evaluation also retains measured envelope excess; completing a time limit does not imply satisfying the MPC’s soft state constraints or the lateral envelope for direct SAC. These diagnostics are sampled at recorded control instants. Direct SAC is evaluated without a speed or lateral state-constraint filter; a fast run with excesses is not an admissible performance under the stated envelope.

This is a reduced-budget recreation, not a quantitative replication. Implementation choices include the specified Deep Sets architecture, observation encoding, periodic circuit training, heterogeneous controller envelopes, 0.2 s/25-interval discretization, solver setup, soft penalties, conservative constant road widths, collision envelopes and reward scaling. The experiment does not establish recovery of the paper’s defensive strategy or its reported safety guarantees.

Recorded solve times depend on local CPU load. Evaluation batches may run concurrently, so these wall times are diagnostics rather than a hardware timing benchmark.
