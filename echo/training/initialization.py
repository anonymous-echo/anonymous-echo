"""Experiments start from the weights of the locally trained step-10000 init run."""

from pathlib import Path

from echo.io import read_json


def verify_initialization(checkpoint, specification):
    """Return the model SHA-256 recorded by the completed init checkpoint."""
    if specification != dict(kind="local", step=10000):
        raise ValueError("Unsupported initialization")
    from echo.training.checkpoint import verify

    parent = Path(checkpoint).parents[1]
    complete = verify(parent)
    config = read_json(parent / "training.json")["identity"]["config"]
    if (
        complete["step"] != 10000
        or config["method"] != "baseline"
        or config["max_steps"] != 10000
    ):
        raise ValueError("Expected the completed baseline init run (step 10000)")
    return complete["files"]["fp32/model.safetensors"]["sha256"]
