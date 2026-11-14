"""Fetch and verify the exact 4,803-scene benchmark used by current experiments."""

import argparse
import gzip
import hashlib

from echo.data.fetch import fetch_source
from echo.io import inside, read_json, sha256

MANIFEST_SHA = "5b1b6709cfd7455f4a6f04ce1bc5107bc37157dd7faa804b2450a299abb98d19"
PROMPT_SHA = "67c21fc2571777c9a61549e548bc27c666b65d65d4c6aa82afbd995693d7a470"
REVISION = "95acf98dd41a4965018069e7e4f27aac1fc69328"


def verify(
    directory,
    manifest="manifests/evaluation_current.json.gz",
    prompts="manifests/prompts_current.json",
):
    with gzip.open(manifest, "rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA or sha256(prompts) != PROMPT_SHA:
        raise ValueError("Current benchmark manifest or prompt registry differs")
    content = read_json(manifest)
    jobs = content["jobs"]
    if (
        content["dataset_revision"] != REVISION
        or len(jobs) != 48030
        or len({j["id"] for j in jobs}) != 48030
    ):
        raise ValueError("Current benchmark identity differs")
    scenes = {j["path"]: j for j in jobs}
    if len(scenes) != 4803:
        raise ValueError("Expected 4803 scenes")
    for name, job in scenes.items():
        scene = inside(directory, name)
        state = scene.with_name(scene.stem + "_sim_state.npy")
        if sha256(scene) != job["bddl_sha256"] or sha256(state) != job["state_sha256"]:
            raise ValueError("Scene/snapshot differs: " + name)
    return dict(
        scenes=4803,
        trials=48030,
        manifest_sha256=MANIFEST_SHA,
        prompt_sha256=PROMPT_SHA,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/failure_scenes")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        fetch_source(
            dict(
                provider="modelscope",
                repo_id="ataier/LIBERO_Recovery_FailureScenes",
                revision=REVISION,
                local_dir=args.directory,
            )
        )
    print(verify(args.directory))


if __name__ == "__main__":
    main()
