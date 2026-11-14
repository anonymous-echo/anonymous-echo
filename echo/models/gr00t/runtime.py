"""Pinned N1.5 runtime; frozen VLM BF16, trainable parameters FP32."""

import os
from pathlib import Path
import random
import sys

from echo.io import read_json, sha256

UPSTREAM = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
PUBLIC_REVISION = "869830fc749c35f34771aa5209f923ac57e4564e"


def setup(config):
    source = Path(config["upstream"]).resolve()
    receipt = read_json(
        Path(__file__).resolve().parents[3]
        / "assets/runtime/gr00t_upstream.json"
    )
    if receipt["revision"] != UPSTREAM:
        raise ValueError("GR00T source receipt revision differs")
    expected = {
        name: record
        for name, record in receipt["files"].items()
        if name.startswith("gr00t/")
    }
    package = source / "gr00t"
    if not expected or not package.is_dir():
        raise ValueError("Missing pinned GR00T source package")
    actual = {
        str(path.relative_to(source))
        for path in package.rglob("*")
        if (path.is_file() or path.is_symlink()) and "__pycache__" not in path.parts
    }
    if actual != set(expected):
        raise ValueError("GR00T source package file inventory differs")
    for name, record in expected.items():
        path = source / name
        if (
            path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError("GR00T source package changed: " + name)
    sys.path.insert(0, str(source))
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        NO_ALBUMENTATIONS_UPDATE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    return UPSTREAM


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def public_inventory(config):
    directory = Path(config["public_checkpoint"])
    expected = read_json(
        Path(__file__).resolve().parents[3] / "assets/runtime/gr00t_runtime.json"
    )
    if expected["model_revision"] != PUBLIC_REVISION:
        raise ValueError("Pinned public model receipt differs")
    for name, record in expected["files"].items():
        from echo.io import inside

        path = inside(directory, name)
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Public model changed: " + name)
    return dict(
        repository="nvidia/GR00T-N1.5-3B",
        revision=PUBLIC_REVISION,
        files=expected["files"],
    )


def load_model(config, device, initialization=None, from_baseline=False):
    import torch
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from echo.models.gr00t.model import BaselineGR00T, EchoGR00T, IsolatedSignalGR00T

    base = GR00T_N1_5.from_pretrained(
        str(config["public_checkpoint"]),
        torch_dtype=torch.bfloat16,
        tune_visual=False,
        tune_llm=False,
        tune_projector=True,
        tune_diffusion_model=True,
        local_files_only=True,
    )
    head = base.action_head
    head.beta_dist = torch.distributions.Beta(
        torch.tensor(head.config.noise_beta_alpha, dtype=torch.float32),
        torch.tensor(head.config.noise_beta_beta, dtype=torch.float32),
    )
    for parameter in base.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    method = config["method"]
    if method == "echo":
        model = EchoGR00T(base, config["forcing"])
    elif method == "isolated":
        model = IsolatedSignalGR00T(base, config["forcing"])
    elif method in {"baseline", "baseline_failure"}:
        model = BaselineGR00T(base)
    else:
        raise ValueError("Unknown GR00T method")
    if initialization is not None:
        from safetensors.torch import load_file

        values = load_file(str(Path(initialization) / "model.safetensors"))
        missing, unexpected = model.load_state_dict(values, strict=False)
        # A baseline initialization lacks only the signal parameters.
        prefixes = {
            "echo": ("ss_projection.", "fs_projection."),
            "isolated": ("status.",),
        }.get(method, ())
        expected = (
            {k for k in model.state_dict() if k.startswith(prefixes)}
            if from_baseline and prefixes
            else set()
        )
        if set(missing) != expected or unexpected:
            raise ValueError(f"Unexpected initialization keys: {missing}, {unexpected}")
        if from_baseline and method == "isolated":
            # The status network starts from the same values as the action head.
            model.status.copy_from(base.action_head)
    return model.to(device)
