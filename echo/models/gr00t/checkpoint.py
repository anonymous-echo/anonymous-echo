"""Atomic native mixed-precision checkpoints with rank RNG and optimizer state."""

from pathlib import Path
import uuid

import torch

from echo.io import atomic_json, inside, read_json, sha256
from echo.training.checkpoint import capture_rng, cpu_tree, optimizer_partition
from echo.training.distributed import all_rank_call, rank_zero_call


def verify(path):
    path = Path(path)
    complete = read_json(path / "complete.json")
    state = read_json(path / "training.json")
    world = state["world_size"]
    if (
        type(world) is not int
        or world < 1
        or type(state["step"]) is not int
        or state["step"] < 1
    ):
        raise ValueError("Invalid GR00T checkpoint step/world size")
    names = {"model.safetensors", "training.json", "statistics.json"} | {
        f"rank_{i:02d}.pt" for i in range(world)
    }
    if (
        complete["format"] != "echo-gr00t-v1"
        or complete["step"] != state["step"]
        or set(complete["files"]) != names
    ):
        raise ValueError("Incomplete GR00T checkpoint inventory")
    if {p.name for p in path.iterdir()} != names | {"complete.json"}:
        raise ValueError("Unexpected GR00T checkpoint files")
    for name, expected in complete["files"].items():
        item = inside(path, name)
        if (
            item.stat().st_size != expected["bytes"]
            or sha256(item) != expected["sha256"]
        ):
            raise ValueError("Checkpoint checksum differs: " + name)
    return complete


def save(run, model, optimizer, step, identity, rank, world):
    from safetensors.torch import save_file

    target = Path(run) / "checkpoints" / f"step_{step:08d}"

    def prepare():
        if target.exists():
            raise FileExistsError(target)
        stage = target.with_name(".pending-" + target.name + "-" + uuid.uuid4().hex)
        stage.mkdir(parents=True)
        return str(stage)

    stage = Path(rank_zero_call(prepare, world))

    def rank_state():
        torch.save(
            dict(
                step=step,
                identity=identity,
                world_size=world,
                rank=rank,
                optimizer=cpu_tree(optimizer.state_dict()),
                rng=capture_rng(getattr(model, "status_stream", None)),
                partition=optimizer_partition(optimizer, model),
            ),
            stage / f"rank_{rank:02d}.pt",
        )

    all_rank_call(rank_state, world)

    def commit():
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            str(stage / "model.safetensors"),
        )
        atomic_json(
            stage / "statistics.json", read_json(identity["config"]["statistics"])
        )
        atomic_json(
            stage / "training.json",
            dict(step=step, world_size=world, identity=identity),
        )
        files = {
            p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
            for p in stage.iterdir()
        }
        atomic_json(
            stage / "complete.json",
            dict(format="echo-gr00t-v1", step=step, files=files),
        )
        stage.rename(target)
        atomic_json(
            Path(run) / "latest.json",
            dict(
                checkpoint=str(target.relative_to(run)),
                step=step,
                complete_sha256=sha256(target / "complete.json"),
            ),
        )

    rank_zero_call(commit, world)


def resume(path, model, optimizer, identity, rank, world):
    from safetensors.torch import load_file

    complete = verify(path)
    metadata = read_json(Path(path) / "training.json")
    if metadata["identity"] != identity or metadata["world_size"] != world:
        raise ValueError("Resume inputs/world size changed")
    model.load_state_dict(load_file(str(Path(path) / "model.safetensors")), strict=True)
    state = torch.load(
        Path(path) / f"rank_{rank:02d}.pt", map_location="cpu", weights_only=False
    )
    if (
        state["identity"] != identity
        or state["step"] != complete["step"]
        or state["rank"] != rank
        or state["world_size"] != world
        or state["partition"] != optimizer_partition(optimizer, model)
    ):
        raise ValueError("Rank checkpoint identity changed")
    optimizer.load_state_dict(state["optimizer"])
    return complete["step"], state["rng"]
