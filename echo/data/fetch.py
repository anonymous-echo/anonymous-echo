"""Download the pinned training data and the public PI05 initialization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

from echo.io import atomic_json, read_json, sha256


def fetch_source(source):
    target = Path(source["local_dir"])
    provider, repository, revision = (
        source["provider"],
        source["repo_id"],
        source["revision"],
    )
    if not repository or not revision:
        raise ValueError(
            "This input has no published repository/revision; configure the real anonymous release first"
        )
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin a complete 40-character revision, not main/HEAD")
    if provider == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repository, repo_type="dataset", revision=revision, local_dir=target
        )
        if source.get("format") == "echo_failure_shards_v1":
            from echo.data.failure_archive import unpack_failure

            unpack_failure(target)
    elif provider == "modelscope":
        if not (target / ".git").exists():
            if target.exists() and any(target.iterdir()):
                raise FileExistsError("Refuse cloning into a nonempty directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://www.modelscope.cn/datasets/" + repository + ".git",
                    str(target),
                ],
                env=env,
                check=True,
            )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", revision], check=True
        )
        actual = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise ValueError("ModelScope revision differs")
        subprocess.run(
            ["git", "-C", str(target), "lfs", "install", "--local"], check=True
        )
        subprocess.run(["git", "-C", str(target), "lfs", "pull"], check=True)
    else:
        raise ValueError("Unknown provider")
    atomic_json(
        target / "download_receipt.json",
        {"provider": provider, "repo_id": repository, "revision": revision},
    )


def fetch_weights(config):
    from huggingface_hub import snapshot_download

    weights = config["weights"]
    snapshot_download(
        weights["repo_id"],
        revision=weights["revision"],
        local_dir=weights["local_dir"],
        allow_patterns=[
            weights["subdir"] + "/fp32/model.safetensors",
            weights["subdir"] + "/config.yaml",
            weights["subdir"] + "/dataset_statistics.json",
        ],
    )
    model = Path(weights["local_dir"]) / weights["subdir"] / "fp32/model.safetensors"
    if sha256(model) != weights["sha256"]:
        raise ValueError("Public PI05 weight checksum differs")
    tokenizer = config["tokenizer"]
    target = Path(tokenizer["local_file"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(".download")
        with (
            urllib.request.urlopen(tokenizer["url"], timeout=60) as response,
            temporary.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        if sha256(temporary) != tokenizer["sha256"]:
            raise ValueError("Tokenizer checksum differs")
        os.replace(temporary, target)
    if sha256(target) != tokenizer["sha256"]:
        raise ValueError("Tokenizer checksum differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.json")
    parser.add_argument(
        "--source",
        choices=[
            "success",
            "success90",
            "recovery",
            "failure",
            "weights",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()
    config = read_json(args.config)
    names = [*config["sources"], "weights"] if args.source == "all" else [args.source]
    for name in names:
        if name == "weights":
            fetch_weights(config)
        else:
            fetch_source(config["sources"][name])


if __name__ == "__main__":
    main()
