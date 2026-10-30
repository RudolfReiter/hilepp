# Checkpoint provenance

`deepsets/` contains the newly trained Deep Sets actors used by the circuit page and default evaluation commands. Each JSON file records architecture, hashes, limits and actual training interactions.

`circuit/` and `pretraining/` are the earlier flat-MLP experiment and are retained for historical inspection only. They are not the current HILEPP model and are rejected by the current circuit evaluator.

Other root-level checkpoints belong to the explicit legacy short-road examples. Full local `*-training.pt` files contain optimizer state and are ignored by Git.
