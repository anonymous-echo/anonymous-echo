"""Native N1.5 DDP training (init run and 2,500-step runs); torchrun is the only launcher."""

from __future__ import annotations

if __name__ == "__main__":
    from echo.runtime import bootstrap_training

    bootstrap_training("gr00t")

import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from echo.data.sampling import EchoWeightedSampler
from echo.models.gr00t.checkpoint import resume, save, verify
from echo.models.gr00t.data import NativeDataset, native_collator
from echo.models.gr00t.runtime import load_model, public_inventory, seed_all, setup
from echo.io import atomic_json, read_json, sha256, source_hashes
from echo.training.checkpoint import restore_rng
from echo.training.distributed import all_rank_call, rank_zero_call


METHODS = {"baseline", "baseline_failure", "echo", "isolated"}
SIGNAL_METHODS = {"echo", "isolated"}


def learning_rate(step, cfg):
    if step <= cfg["warmup_steps"]:
        return cfg["lr"] * step / cfg["warmup_steps"]
    if cfg["lr_schedule"] == "constant":
        return cfg["lr"]
    fraction = (step - cfg["warmup_steps"]) / (cfg["max_steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (
        1 + math.cos(math.pi * fraction)
    )


class DrawSampler:
    def __init__(self, dataset, cfg, world, rank, start_step):
        self.dataset, self.cfg, self.world, self.rank, self.start_step = (
            dataset,
            cfg,
            world,
            rank,
            start_step,
        )
        self.local_batch = cfg["global_batch_size"] // world
        if 320000 % cfg["global_batch_size"]:
            raise ValueError("The draw epoch must divide the global batch")
        self.per_epoch = 320000 // cfg["global_batch_size"]
        self.sampler = EchoWeightedSampler(
            dataset,
            world,
            rank,
            cfg["seed"],
            320000,
            cfg["sampling_weights"],
            32,
            data_profile="baseline" if cfg["method"] == "baseline" else "with_failure",
        )

    def __iter__(self):
        previous = None
        for step in range(self.start_step, self.cfg["max_steps"]):
            epoch, offset = divmod(step, self.per_epoch)
            if epoch != previous:
                self.sampler.set_epoch(epoch)
                previous = epoch
            for j in range(self.local_batch):
                draw = offset * self.local_batch + j
                sequence = (epoch, self.rank, draw, self.cfg["seed"])
                seed = int.from_bytes(
                    hashlib.blake2b(repr(sequence).encode(), digest_size=8).digest(),
                    "little",
                ) % (2**63)
                yield self.sampler.indices[draw], seed

    def __len__(self):
        return (self.cfg["max_steps"] - self.start_step) * self.local_batch


def validate_config(cfg):
    required = {
        "method",
        "upstream",
        "public_checkpoint",
        "initialization",
        "manifest",
        "data_config",
        "statistics",
        "validation_receipt",
        "run_dir",
        "seed",
        "world_size",
        "global_batch_size",
        "micro_batch_size",
        "num_workers",
        "cpu_threads",
        "max_steps",
        "warmup_steps",
        "lr",
        "min_lr",
        "lr_schedule",
        "weight_decay",
        "betas",
        "eps",
        "grad_clip",
        "save_interval",
        "sampling_weights",
        "boundary_radius_seconds",
        "boundary_seed",
        "forcing",
    }
    if set(cfg) != required or cfg["method"] not in METHODS:
        raise ValueError("Unknown GR00T training config")
    if (
        cfg["world_size"] < 1
        or cfg["global_batch_size"] != 128
        or cfg["micro_batch_size"] != 4
    ):
        raise ValueError("Native batch must be global 128 / micro 4")
    if cfg["cpu_threads"] != 4 or cfg["num_workers"] != 2:
        raise ValueError("Native execution uses four CPU threads and two data workers")
    if cfg["global_batch_size"] % (cfg["world_size"] * cfg["micro_batch_size"]):
        raise ValueError("Global batch must divide world*microbatch")
    if (
        cfg["max_steps"] not in (10000, 2500)
        or cfg["seed"] != 42
        or cfg["warmup_steps"] != (1500 if cfg["max_steps"] == 10000 else 125)
    ):
        raise ValueError("GR00T training schedule changed")
    if cfg["lr_schedule"] != ("constant" if cfg["max_steps"] == 10000 else "cosine"):
        raise ValueError("Incorrect stage schedule")
    if cfg["forcing"] != (0.7 if cfg["method"] in SIGNAL_METHODS else 0):
        raise ValueError("Incorrect forcing")
    if cfg["boundary_radius_seconds"] and (
        cfg["method"] != "echo" or cfg["boundary_radius_seconds"] != 0.2
    ):
        raise ValueError("Only the Echo boundary run uses boundary offsets (0.2 s)")
    if (
        cfg["max_steps"] == 10000
        and cfg["initialization"] is not None
        or cfg["max_steps"] == 2500
        and cfg["initialization"] is None
    ):
        raise ValueError("The init run starts from public weights; other runs start from the init run")
    if cfg["max_steps"] == 10000 and cfg["method"] != "baseline":
        raise ValueError("The init run is the success/recovery baseline")
    expected_weights = {"success": 40, "recovery": 20}
    if cfg["method"] != "baseline":
        expected_weights["failure"] = 1
    if cfg["sampling_weights"] != expected_weights:
        raise ValueError("Sampling weights differ")
    if (
        cfg["lr"] != 1e-4
        or cfg["min_lr"] != (1e-4 if cfg["max_steps"] == 10000 else 1e-5)
        or cfg["betas"] != [0.95, 0.999]
        or cfg["eps"] != 1e-8
        or cfg["weight_decay"] != 1e-5
        or cfg["grad_clip"] != 1.0
        or cfg["boundary_seed"] != 20261108
    ):
        raise ValueError("Native optimizer/boundary settings differ")
    return cfg


def counts(batch, method):
    """Global loss denominators: one for all terms, or actions then signals."""
    action = batch["action_mask"].sum()
    signal = batch["ss_mask"].sum() + batch["fs_mask"].sum() if method in SIGNAL_METHODS else 0
    if method == "isolated":
        return torch.stack((action, signal))
    return torch.stack((action + signal,))


def make_dataset(cfg):
    setup(cfg)
    return NativeDataset(
        cfg["manifest"],
        cfg["data_config"],
        read_json(cfg["statistics"]),
        method=cfg["method"],
        boundary_radius_seconds=cfg["boundary_radius_seconds"],
        boundary_seed=cfg["boundary_seed"],
    )


def preflight(cfg):
    validate_config(cfg)
    from echo.runtime import verify_runtime

    verify_runtime("native")
    # Preflight is rank-zero only during training; every rank still needs the
    # pinned upstream import path before constructing its model and processor.
    setup(cfg)
    if cfg["max_steps"] == 2500:
        initial = Path(cfg["initialization"])
        verify(initial)
        parent = read_json(initial / "training.json")["identity"]["config"]
        if (
            read_json(initial / "complete.json")["step"] != 10000
            or parent["method"] != "baseline"
            or parent["max_steps"] != 10000
        ):
            raise ValueError("Expected the completed baseline init run (step 10000)")
        if read_json(initial / "statistics.json") != read_json(cfg["statistics"]):
            raise ValueError("Init normalization differs")
    public = public_inventory(cfg)
    statistics = read_json(cfg["statistics"])
    if (
        statistics["normalization"] != "native_min_max"
        or statistics["pool"] != "unpadded_success_recovery_only"
    ):
        raise ValueError("GR00T statistics differ")
    from echo.data.validation import verify_receipt

    options = dict(horizon=16, raw=True)
    if cfg["boundary_radius_seconds"]:
        options.update(
            boundary_radius_seconds=cfg["boundary_radius_seconds"],
            boundary_seed=cfg["boundary_seed"],
        )
    verify_receipt(
        cfg["validation_receipt"],
        cfg["manifest"],
        cfg["data_config"],
        cfg["statistics"],
        cfg["method"],
        **options,
    )
    dataset = make_dataset(cfg)
    return dataset, dict(
        config=cfg,
        manifest_sha256=sha256(cfg["manifest"]),
        data_config_sha256=sha256(cfg["data_config"]),
        statistics_sha256=sha256(cfg["statistics"]),
        initialization_sha256=sha256(Path(cfg["initialization"]) / "model.safetensors")
        if cfg["initialization"]
        else None,
        validation_sha256=sha256(cfg["validation_receipt"]),
        public=public,
        sources=source_hashes(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Latest complete checkpoint directory")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    cfg = read_json(args.config)
    validate_config(cfg)
    from echo.runtime import verify_runtime

    runtime_receipt = verify_runtime("native")
    torch.set_num_threads(cfg["cpu_threads"])
    if args.preflight:
        dataset, identity = preflight(cfg)
        print(
            json.dumps(dict(preflight="passed", frames=len(dataset), identity=identity))
        )
        return
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if (
        world != cfg["world_size"]
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("GR00T training needs configured CUDA BF16 ranks")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    identity = rank_zero_call(lambda: preflight(cfg)[1], world)
    identity["runtime"] = runtime_receipt
    run = Path(cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / f"rank_{rank:02d}.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def check_run():
            path = run / "identity.json"
            if path.exists():
                if read_json(path) != identity or not args.resume:
                    raise ValueError(
                        "Existing run identity differs or requires --resume"
                    )
            elif args.resume:
                raise ValueError("Resume requires existing identity")
            else:
                atomic_json(path, identity)

        rank_zero_call(check_run, world)
        seed_all(cfg["seed"])
        model = load_model(
            cfg,
            f"cuda:{local}",
            cfg["initialization"],
            from_baseline=cfg["method"] in SIGNAL_METHODS
            and cfg["initialization"] is not None,
        )
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        isolated = cfg["method"] == "isolated"
        groups = [[p for _, p in named]]
        if isolated:
            # Each network is clipped by its own norm, so they never couple.
            groups = [
                [p for n, p in named if n.startswith(prefix)]
                for prefix in ("base.", "status.")
            ]
            if sum(map(len, groups)) != len(named) or not all(groups):
                raise ValueError("Each parameter must belong to exactly one network")
        decay = [p for n, p in named if p.ndim > 1 and "norm" not in n.lower()]
        no_decay = [p for n, p in named if p.ndim <= 1 or "norm" in n.lower()]
        optimizer = torch.optim.AdamW(
            [
                dict(params=decay, weight_decay=cfg["weight_decay"]),
                dict(params=no_decay, weight_decay=0.0),
            ],
            lr=cfg["lr"],
            betas=tuple(cfg["betas"]),
            eps=cfg["eps"],
            foreach=False,
        )
        start = 0
        pending_rng = None
        if args.resume:
            start, pending_rng = all_rank_call(
                lambda: resume(args.resume, model, optimizer, identity, rank, world),
                world,
            )
            latest = read_json(run / "latest.json")
            if (
                Path(args.resume).resolve() != (run / latest["checkpoint"]).resolve()
                or sha256(Path(args.resume) / "complete.json")
                != latest["complete_sha256"]
            ):
                raise ValueError("Resume is not this run latest complete checkpoint")
        end = (
            min(cfg["max_steps"], args.stop_after)
            if args.stop_after
            else cfg["max_steps"]
        )
        if not start < end <= cfg["max_steps"]:
            raise ValueError("Invalid stop-after")
        ddp = (
            DistributedDataParallel(
                model,
                device_ids=[local],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )
            if world > 1
            else model
        )
        seed_all(cfg["seed"] + rank)
        stream = getattr(model, "status_stream", None)
        if stream is not None:
            stream.seed(cfg["seed"] + rank, f"cuda:{local}")
        dataset = all_rank_call(lambda: make_dataset(cfg), world)
        sampler = DrawSampler(dataset, cfg, world, rank, start)
        accum = cfg["global_batch_size"] // world // cfg["micro_batch_size"]
        loader = DataLoader(
            dataset,
            batch_size=cfg["micro_batch_size"],
            sampler=sampler,
            num_workers=cfg["num_workers"],
            collate_fn=native_collator(),
            generator=torch.Generator().manual_seed(cfg["seed"] + rank),
            multiprocessing_context="spawn" if cfg["num_workers"] else None,
            persistent_workers=bool(cfg["num_workers"]),
        )
        iterator = iter(loader)
        ddp.train()
        if pending_rng:
            restore_rng(pending_rng, stream)
        for step in range(start + 1, end + 1):
            batches = [next(iterator) for _ in range(accum)]
            denominators = sum(counts(batch, cfg["method"]) for batch in batches).to(
                f"cuda:{local}"
            )
            if world > 1:
                dist.all_reduce(denominators)
            if (denominators <= 0).any():
                raise ValueError("No supervised entries")
            lr = learning_rate(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            numerators = [0.0] * len(denominators)
            for index, batch in enumerate(batches):
                batch.pop("audit_type")
                batch.pop("audit_padding")
                sync = (
                    ddp.no_sync() if world > 1 and index < accum - 1 else nullcontext()
                )
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    result = ddp(batch)
                    terms = (
                        [result["action_numerator"], result["status_numerator"]]
                        if isolated
                        else [result["loss_numerator"]]
                    )
                    # The two networks have disjoint parameters, so each term
                    # is normalized by its own global count.
                    loss = sum(t * world / d for t, d in zip(terms, denominators))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite GR00T loss")
                loss.backward()
                for i, term in enumerate(terms):
                    numerators[i] += float(term.detach())
            for parameters in groups:
                torch.nn.utils.clip_grad_norm_(
                    parameters, cfg["grad_clip"], error_if_nonfinite=True
                )
            optimizer.step()
            if rank == 0:
                record = dict(
                    step=step,
                    lr=lr,
                    loss_numerator=numerators,
                    loss_denominator=[float(d) for d in denominators],
                    unix=time.time(),
                )
                with (run / "loss.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                atomic_json(run / "progress.json", record)
                if step % 10 == 0:
                    print(json.dumps(record), flush=True)
            if step == 10 or step % cfg["save_interval"] == 0 or step == end:
                save(run, model, optimizer, step, identity, rank, world)
        rank_zero_call(
            lambda: atomic_json(
                run / "progress.json",
                dict(
                    step=end, stage="complete" if end == cfg["max_steps"] else "stopped"
                ),
            ),
            world,
        )
    finally:
        lock.close()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
