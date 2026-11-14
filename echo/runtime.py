"""Fail before a run if its numerical environment differs from the audited one."""

from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys


def bootstrap_training(backbone):
    """Apply the pinned launcher settings before NumPy/PyTorch imports."""
    threads = "8" if backbone == "pi05" else "4"
    required = dict(
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONNOUSERSITE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    if backbone == "gr00t":
        required["PYTHONHASHSEED"] = "0"
    if any(os.environ.get(key) != value for key, value in required.items()):
        environment = os.environ.copy()
        environment.update(required)
        module = "echo.training.train" if backbone == "pi05" else "echo.models.gr00t.train"
        os.execve(
            sys.executable,
            [sys.executable, "-B", "-m", module, *sys.argv[1:]],
            environment,
        )


def verify_runtime(profile):
    path = (
        Path(__file__).resolve().parents[1]
        / "assets/runtime/environments.json"
    )
    expected = json.loads(path.read_text())[profile]
    mismatches = {}
    if platform.system() != "Linux":
        mismatches["platform"] = dict(expected="Linux", actual=platform.system())
    if platform.python_version() != expected["python"]:
        mismatches["python"] = dict(
            expected=expected["python"], actual=platform.python_version()
        )
    actual = {}
    for name, required in expected["packages"].items():
        try:
            actual[name] = version(name)
        except PackageNotFoundError:
            actual[name] = None
        if actual[name] != required:
            mismatches[name] = dict(expected=required, actual=actual[name])
    if mismatches:
        raise RuntimeError(f"Pinned {profile} environment differs: {mismatches}")
    return dict(profile=profile, python=platform.python_version(), packages=actual)


def verify_simulator_sources():
    from echo.io import sha256

    root = Path(__file__).resolve().parents[1]
    expected = json.loads(
        (root / "assets/runtime/simulator_sources.json").read_text()
    )["sha256"]
    actual = {
        str(p.relative_to(root))
        for prefix in ("LIBERO/libero", "starVLA/model")
        for p in (root / prefix).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    }
    if actual != set(expected):
        raise RuntimeError("Frozen LIBERO/StarVLA source inventory differs")
    for name, checksum in expected.items():
        if sha256(root / name) != checksum:
            raise RuntimeError("Frozen LIBERO/StarVLA source changed: " + name)
    return dict(
        files=len(expected),
        inventory_sha256=sha256(root / "assets/runtime/simulator_sources.json"),
    )
