"""Validate the actual published Deep Sets weights and their recorded provenance."""

import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from hilepp.policy import NumpyPolicy, NumpyGaussianPolicy

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/data/manifest.json").read_text())
for method in ["HILEPP", "SAC"]:
    metadata = manifest["training"][method]
    assert metadata["architecture"] == "deep-sets-sum-v1"
    assert metadata["encoder"]["pooling"] == "sum"
    prefix = root / "checkpoints/deepsets" / method.lower()
    mean_file = prefix.with_suffix(".npz")
    full_file = Path(str(prefix) + "-distribution.npz")
    assert hashlib.sha256(mean_file.read_bytes()).hexdigest() == metadata["checkpoint_sha256"]
    assert hashlib.sha256(full_file.read_bytes()).hexdigest() == metadata["distribution_sha256"]
    policy = NumpyPolicy(mean_file)
    assert policy.deep_sets
    with np.load(str(prefix) + "-initial.npz", allow_pickle=False) as initial:
        for layer in ["encoder.phi.0.weight", "encoder.rho.0.weight"]:
            assert np.max(abs(initial[layer] - policy.parameters[layer])) > 1e-6
    with np.load(full_file, allow_pickle=False) as full:
        for name, values in policy.parameters.items():
            np.testing.assert_array_equal(values, full[name])
    for track in dict.fromkeys(r["track"] for r in manifest["runs"]):
        entry = next(r for r in manifest["runs"] if r["track"] == track and r["method"] == method)
        run = json.loads(gzip.decompress((root / "docs/data" / entry["file"]).read_bytes()))
        assert [v["limits"] for v in run["vehicles"]] == metadata["vehicle_limits"]
        geometry = run["track"]
        for index in np.linspace(0, len(run["states"]) - 1, 8).astype(int):
            state = np.asarray(run["states"][index])
            ego, others = state[0], state[1:].copy()
            others[:, 0] -= ego[0]
            others[:, 0] -= np.round(others[:, 0] / geometry["length"]) * geometry["length"]
            curvature = (
                np.interp(
                    (ego[0] + np.array([0, 10, 25, 50, 80, 120, 180])) % geometry["length"],
                    geometry["s"],
                    geometry["curvature"],
                )
                / 0.05
            )
            normalized = others[:, [0, 1, 3, 2]] / [120, geometry["half_width"], 40, 0.45]
            context = np.r_[curvature, ego[[1, 3, 2, 4]] / [geometry["half_width"], 40, 0.45, 0.3]]
            first = np.r_[context, normalized.ravel()]
            second = np.r_[context, normalized[[3, 0, 4, 2, 1]].ravel()]
            np.testing.assert_allclose(policy(first), policy(second), atol=2e-6)
            a, b = NumpyGaussianPolicy(full_file, 73), NumpyGaussianPolicy(full_file, 73)
            np.testing.assert_allclose(a(first), b(second), atol=2e-6)
    print(f"{method}: learned encoder, published hashes and recorded-state permutation checks pass")
