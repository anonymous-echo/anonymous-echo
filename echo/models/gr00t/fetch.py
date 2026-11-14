"""Download pinned GR00T weights and the Isaac-GR00T source package."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/pretrained/gr00t_public")
    parser.add_argument("--upstream", help="Clone Isaac-GR00T into this directory")
    args = parser.parse_args()
    if sys.platform == "darwin":
        raise SystemExit(
            "Download directly on the Linux training/development host, not through the Mac"
        )
    if args.upstream:
        import subprocess
        from echo.models.gr00t.runtime import UPSTREAM

        target = Path(args.upstream)
        target.parent.mkdir(parents=True, exist_ok=True)
        if (target / ".git").exists():
            subprocess.run(["git", "-C", str(target), "fetch"], check=True)
        else:
            subprocess.run(
                ["git", "clone", "https://github.com/NVIDIA/Isaac-GR00T.git", str(target)],
                check=True,
            )
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", UPSTREAM], check=True)
        from echo.models.gr00t.runtime import setup

        setup({"upstream": str(target)})
        print(f"Isaac-GR00T checked out at {UPSTREAM} and verified")
    from huggingface_hub import snapshot_download
    from echo.models.gr00t.runtime import PUBLIC_REVISION, public_inventory

    snapshot_download(
        "nvidia/GR00T-N1.5-3B",
        revision=PUBLIC_REVISION,
        local_dir=args.directory,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "experiment_cfg/metadata.json",
            "LICENSE",
            "README.md",
        ],
    )
    public_inventory({"public_checkpoint": args.directory})
    print("Pinned public GR00T inputs verified")


if __name__ == "__main__":
    main()
