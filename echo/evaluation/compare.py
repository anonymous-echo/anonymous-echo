"""Scene-clustered comparison of two complete evaluations: difference, CI and p-values.

Trials of one scene are correlated, so the scene is the unit of resampling. For
each scene, sum the run-minus-baseline success differences over its trials.
- 95% CI: scene bootstrap of the pooled difference (4,000 resamples).
- p (sign-flip): two-sided paired randomization test that flips the sign of
  every scene total; 1,000,000 random flips, reported as (k + 1) / (B + 1).
- p (cluster z): normal approximation with the scene-level standard error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from echo.evaluation.results import inspect_results
from echo.io import atomic_json, read_json

SEED = 20261113


def scene_totals(rows, baseline, jobs):
    totals, sizes = {}, {}
    for job in jobs:
        scene = job["scene_id"]
        difference = int(rows[job["id"]]["success"]) - int(
            baseline[job["id"]]["success"]
        )
        totals[scene] = totals.get(scene, 0) + difference
        sizes[scene] = sizes.get(scene, 0) + 1
    return (
        np.array(list(totals.values()), dtype=np.int64),
        np.array([sizes[s] for s in totals], dtype=np.int64),
    )


def scene_test(values, sizes, seed=SEED, resamples=4000, flips=1_000_000):
    scenes, trials, observed = len(values), int(sizes.sum()), int(values.sum())
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for i in range(resamples):
        chosen = rng.integers(scenes, size=scenes)
        boot[i] = values[chosen].sum() / sizes[chosen].sum()
    rng = np.random.default_rng(seed)
    chunk = min(10_000, flips)
    if flips % chunk:
        raise ValueError("flips must be a multiple of the chunk size")
    extreme = 0
    for _ in range(flips // chunk):
        signs = rng.integers(0, 2, size=(chunk, scenes), dtype=np.int8)
        signs = signs.astype(np.int32) * 2 - 1
        extreme += int(np.sum(np.abs(signs @ values) >= abs(observed)))
    means = values / sizes
    z = means.mean() / (means.std(ddof=1) / math.sqrt(scenes))
    return dict(
        scenes=scenes,
        trials=trials,
        difference_pp=100 * observed / trials,
        scene_bootstrap_95_pp=[100 * x for x in np.quantile(boot, [0.025, 0.975])],
        scenes_run_better=int((values > 0).sum()),
        scenes_baseline_better=int((values < 0).sum()),
        p_sign_flip_two_sided=(extreme + 1) / (flips + 1),
        sign_flips=flips,
        z=float(z),
        p_cluster_z_two_sided=math.erfc(abs(z) / math.sqrt(2)),
    )


def compare(run, baseline, manifest):
    jobs = read_json(manifest)["jobs"]
    reports = {}
    for name, directory in (("run", run), ("baseline", baseline)):
        rows, status = inspect_results(directory, manifest)
        if not status["complete"]:
            raise ValueError(f"{directory} is incomplete or has invalid trials")
        reports[name] = rows
    values, sizes = scene_totals(reports["run"], reports["baseline"], jobs)
    result = dict(
        run=str(run),
        baseline=str(baseline),
        run_successes=sum(r["success"] for r in reports["run"].values()),
        baseline_successes=sum(r["success"] for r in reports["baseline"].values()),
    )
    result.update(scene_test(values, sizes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--manifest", default="manifests/evaluation_current.json.gz")
    parser.add_argument("--output", help="Default: <run>/comparison.json")
    args = parser.parse_args()
    result = compare(args.run, args.baseline, args.manifest)
    atomic_json(args.output or Path(args.run) / "comparison.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
