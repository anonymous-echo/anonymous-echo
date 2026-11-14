"""Numerical contracts for boundary offsets, three-class status and native GR00T ports."""

import importlib.util
import hashlib
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from echo.config import load_training
from echo.data.chunks import ChunkDataset, signal_targets
from echo.data.current_status import current_label
from echo.data.boundary import boundary_offset
from echo.data.sampling import EchoWeightedSampler
from echo.evaluation.evaluate import rollout
from echo.evaluation.signals import signal_accuracy, whole_chunk
from echo.models.gr00t import checkpoint as native_checkpoint, runtime as native_runtime
from echo.models.gr00t.train import DrawSampler, counts, validate_config
from echo.io import atomic_json, read_json
from echo.models.model import EchoModelConfig
from echo.models.status import StatusHeadPI05, current_status_loss
from echo.training.checkpoint import restore_rng

ROOT = Path(__file__).resolve().parents[2]


def native_model_module():
    # The objective wrappers use only Torch. A local stub isolates the optional
    # HF result container; this is not a native-model integration test.
    stub = ModuleType("transformers.feature_extraction_utils")
    stub.BatchFeature = dict
    spec = importlib.util.spec_from_file_location(
        "native_objective_test", ROOT / "echo/models/gr00t/model.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"transformers.feature_extraction_utils": stub}):
        spec.loader.exec_module(module)
    return module


def native_baseline_class():
    return native_model_module().BaselineGR00T


class ExtensionContracts(unittest.TestCase):
    def test_three_class_tail_sampler_matches_frozen_draws(self):
        dataset = SimpleNamespace(
            offsets=[0, 100, 200, 300],
            segments=[
                dict(segment_type=kind) for kind in ("success", "recovery", "failure")
            ],
        )
        expected = (
            "6af23716d1bd39e35f7240b6e66a3a9d3c8097ab4bee8e1d3be06458c25eadd9",
            "6b8fc3dd71b9e79acc7e2a2259c6293ec7170f01d165280f0fe3aa325dc2a2bf",
            "a4b19b1c783151e1b2c79b7354bc15df6ce5705c91e269af4259b11d7de4d263",
            "41c7881e6851cd3d413171c05b79936d060ef4cfd5dd7dbbff7f5f83abbdeee4",
        )
        for rank, digest in enumerate(expected):
            sampler = EchoWeightedSampler(
                dataset,
                num_replicas=4,
                rank=rank,
                seed=42,
                samples_per_epoch=32000,
                weights=dict(success=40, recovery=20, failure=1),
                tail_sample_probability=0.5,
            )
            actual = hashlib.sha256(
                np.asarray(sampler.indices, np.int64).tobytes()
            ).hexdigest()
            self.assertEqual(actual, digest)
            sampler.set_epoch(0, batch_offset=10, batch_size=32)
            self.assertEqual(list(sampler), sampler.indices[320:])

    def test_native_source_receipt_accepts_archive_and_rejects_changes(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            package = source / "gr00t"
            package.mkdir()
            file = package / "__init__.py"
            file.write_text("value = 1\n")
            payload = file.read_bytes()
            receipt = dict(
                revision=native_runtime.UPSTREAM,
                files={
                    "gr00t/__init__.py": dict(
                        bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
                    )
                },
            )
            with (
                patch.object(native_runtime, "read_json", return_value=receipt),
                patch.object(sys, "path", sys.path.copy()),
            ):
                self.assertEqual(
                    native_runtime.setup({"upstream": str(source)}),
                    native_runtime.UPSTREAM,
                )
                file.write_text("value = 2\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    native_runtime.setup({"upstream": str(source)})
                file.write_bytes(payload)
                (package / "extra.py").write_text("")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    native_runtime.setup({"upstream": str(source)})

    def test_boundary_shifts_sampling_padding_and_raw_actions_together(self):
        for horizon in (16, 50):
            for offset in (-3, 3):
                with (
                    self.subTest(horizon=horizon, offset=offset),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    # Find real hash inputs producing each sign; never mock the draw.
                    offset_id = next(
                        f"episode{i}"
                        for i in range(1000)
                        if boundary_offset(f"episode{i}", 20, 0.2, 20261108) == offset
                    )
                    seg = dict(
                        segment_id="portable_id",
                        boundary_offset_id=offset_id,
                        segment_type="success",
                        source="s",
                        boundary=dict(start=0, end=10),
                        source_lang="task",
                        canonical_prompt="task",
                        source_gripper_convention="dataset_binary_01",
                    )
                    atomic_json(
                        root / "manifest.json",
                        dict(schema_version=1, horizon=50, segments=[seg]),
                    )
                    atomic_json(
                        root / "data.json",
                        dict(sources={"s": dict(local_dir=str(root))}),
                    )
                    atomic_json(root / "meta/info.json", dict(fps=20))
                    stats = {
                        k: dict(q01=[-2.0] * n, q99=[3.0] * n)
                        for k, n in [("actions", 7), ("state", 8)]
                    }
                    data = ChunkDataset(
                        root / "manifest.json",
                        root / "data.json",
                        stats,
                        raw=True,
                        load_images=False,
                        horizon=horizon,
                        boundary_radius_seconds=0.2,
                    )
                    states = np.arange(80, dtype=np.float32).reshape(10, 8)
                    actions = np.ones((10, 7), np.float32) * 0.25
                    data._load = lambda _: (states, actions, [])
                    self.assertEqual(len(data), 10 + offset)
                    for frame in range(len(data)):
                        sample = data[frame]
                        expected = np.arange(horizon) >= 10 + offset - frame
                        np.testing.assert_array_equal(
                            sample["echo_ss"], expected.astype(np.float32)
                        )
                        np.testing.assert_array_equal(sample["padding_mask"], expected)
                        source_valid = max(
                            0, min(horizon, 10 - frame, 10 + offset - frame)
                        )
                        np.testing.assert_array_equal(
                            sample["action"][source_valid:], 0
                        )
                        if source_valid:
                            np.testing.assert_array_equal(
                                sample["action"][:source_valid, :6], 0.25
                            )
                            np.testing.assert_array_equal(
                                sample["action"][:source_valid, 6], 0.5
                            )
                        np.testing.assert_array_equal(
                            sample["state"][0], states[min(frame, 9)]
                        )
                    if offset > 0:
                        self.assertFalse(data[10]["padding_mask"][0])
                        self.assertTrue(data[10]["echo_action_mask"].all())
                        # Raw zero is normalized only after extension; a zero in
                        # normalized coordinates would teach a different target.
                        data.raw = False
                        np.testing.assert_allclose(
                            data[10]["action"], 2.0 * 2.0 / (5.0 + 1e-6) - 1.0
                        )
                    with self.assertRaises(IndexError):
                        data[len(data)]

    def test_current_observation_labels_have_exact_five_frame_tails(self):
        for kind in ("success", "recovery", "failure"):
            labels = [current_label(kind, f, 4, 14, 3) for f in range(4, 14)]
            terminal = 1 if kind != "failure" else 2
            self.assertEqual(labels, [0] * 5 + [terminal] * 5)
        with self.assertRaises(ValueError):
            current_label("success", 4, 4, 14, 2)
        self.assertEqual(
            [current_label("recovery", f, 2, 5, 3) for f in range(2, 5)], [1] * 3
        )

    def test_status_failure_has_only_ce_and_recovery_keeps_denominator(self):
        errors = torch.ones(1, 50, 7, requires_grad=True)
        logits = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        mask = torch.zeros_like(errors, dtype=torch.bool)
        bound = torch.ones(1, 50, dtype=torch.bool)
        result = current_status_loss(
            errors, mask, bound, logits, torch.tensor([2]), 1.0
        )
        result["loss"].backward()
        self.assertEqual(result["action_loss"].item(), 0.0)
        self.assertEqual(errors.grad.abs().sum().item(), 0.0)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)
        mask[:] = True
        bound[:, 10:] = False
        result = current_status_loss(
            errors, mask, bound, logits, torch.tensor([2]), 1.0
        )
        self.assertAlmostEqual(result["action_loss"].item(), 0.2)

    def test_sequential_action_gradient_reaches_probability_head(self):
        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_horizon = 50
                self.action_dim = 32
                self.vlm = SimpleNamespace(
                    config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=12))
                )
                self.action_head = nn.Module()
                self.action_head.action_in_proj = nn.Linear(32, 16, bias=False)
                self.action_head.time_mlp_in = nn.Linear(16, 16)
                self.action_head.time_mlp_out = nn.Linear(16, 16)

        for method in ("classifier", "sequential"):
            model = StatusHeadPI05(
                Base(), EchoModelConfig(), method=method, status_classes=3
            )
            logits = model._status_logits(
                torch.randn(2, 4, 12),
                torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
            )
            suffix, _ = model._action_suffix(
                torch.randn(2, 50, 7), logits.softmax(-1), torch.ones(2)
            )
            suffix.square().sum().backward()
            grad = sum(
                p.grad.abs().sum().item()
                for p in model.status_head.parameters()
                if p.grad is not None
            )
            self.assertGreater(
                grad, 0.0
            ) if method == "sequential" else self.assertEqual(grad, 0.0)

    def test_native_failure_sentinel_and_recovery_loss(self):
        class Native(nn.Module):
            def __init__(self):
                super().__init__()
                self.value = nn.Parameter(torch.tensor(2.0))

            def forward(self, inputs):
                mask = inputs["action_mask"]
                return {"loss": (self.value.square() * mask).sum() / mask.sum()}

        model = native_baseline_class()(Native(), gradient_checkpointing=False)
        mask = torch.zeros(2, 16, 32, dtype=torch.bool)
        bound = torch.ones(2, 16, dtype=torch.bool)
        result = model(dict(action_mask=mask, action_bound=bound))
        result["loss_numerator"].backward()
        self.assertEqual(result["loss_numerator"].item(), 0.0)
        self.assertEqual(result["loss_count"].item(), 0)
        self.assertEqual(model.base.value.grad.item(), 0.0)
        self.assertFalse(mask.any())  # sentinel must not mutate the caller
        mask[:, :, :7] = True
        bound[:, 4:] = False
        result = model(dict(action_mask=mask, action_bound=bound))
        self.assertEqual(result["loss_count"].item(), 2 * 16 * 7)
        self.assertAlmostEqual(result["loss"].item(), 1.0)
        self.assertAlmostEqual(result["loss_numerator"].item(), 2 * 4 * 7 * 4)

    def test_native_checkpoint_resumes_random_updates_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "stats.json", {})
            identity = {"config": {"statistics": str(root / "stats.json")}}
            random.seed(4)
            np.random.seed(4)
            torch.manual_seed(4)
            model = nn.Linear(3, 2)
            opt = torch.optim.AdamW(model.parameters(), lr=0.01)

            def update(m, o):
                o.zero_grad()
                x = torch.randn(3, 3) + random.random() + np.random.random()
                m(x).square().mean().backward()
                o.step()

            update(model, opt)
            native_checkpoint.save(root, model, opt, 1, identity, 0, 1)
            update(model, opt)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restored = nn.Linear(3, 2)
            new_opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
            path = root / "checkpoints/step_00000001"
            step, rng = native_checkpoint.resume(
                path, restored, new_opt, identity, 0, 1
            )
            self.assertEqual(step, 1)
            restore_rng(rng)
            update(restored, new_opt)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
            for a, b in zip(opt.state.values(), new_opt.state.values()):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
            (path / "extra").touch()
            with self.assertRaises(ValueError):
                native_checkpoint.verify(path)

    def test_native_draw_resume_preserves_indices_and_augmentation_seed(self):
        dataset = SimpleNamespace(
            segments=[{"segment_type": k} for k in ("success", "failure", "recovery")],
            offsets=[0, 50, 110, 150],
        )
        cfg = dict(
            max_steps=4,
            global_batch_size=128,
            method="echo",
            seed=42,
            sampling_weights={"success": 40, "failure": 1, "recovery": 20},
        )
        full = list(DrawSampler(dataset, cfg, 4, 2, 0))
        self.assertEqual(list(DrawSampler(dataset, cfg, 4, 2, 2)), full[64:])

    def test_h16_rollout_and_categorical_argmax(self):
        obs = {
            k: np.zeros((4, 4, 3), np.uint8)
            for k in ("agentview_image", "robot0_eye_in_hand_image")
        }

        class Env:
            def step(self, action):
                return obs, 0, False, {}

        class Policy:
            action_horizon = 16
            last_signals = None

            def infer(self, *_):
                self.last_signals = {
                    "ss": np.ones(16) * 0.35,
                    "fs": np.ones(16) * 0.25,
                    "status_probabilities": np.array([0.4, 0.35, 0.25]),
                    "status_label": 0,
                }
                return np.zeros((16, 7), np.float32)

        with patch("echo.evaluation.evaluate.state_vector", return_value=np.zeros(8)):
            row = rollout(
                Env(), obs, Policy(), "task", dict(max_steps=20, execute_horizon=8)
            )
        self.assertEqual(row["ss"].shape, (3, 16))
        self.assertEqual(row["status_probabilities"].shape, (3, 3))
        metric = signal_accuracy(
            np.zeros(20, dtype=bool),
            [0, 8, 16],
            [0.4] * 3,
            [0.6] * 3,
            status_labels=np.array([2, 2, 2]),
        )
        self.assertEqual(
            metric["ongoing_frame_correct"], 0
        )  # argmax failure, even SS<.5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.npz"
            np.savez(path, **{k: row[k] for k in ("query_controller_step", "ss", "fs")})
            self.assertEqual(whole_chunk(path, 20)[1].shape, (3,))

    def test_experiment_registry_and_configs(self):
        registry = read_json(ROOT / "configs/experiments.json")
        keys = [x["key"] for x in registry["experiments"]]
        self.assertEqual(
            keys,
            [
                "pi05_init",
                "pi05_baseline",
                "pi05_echo07",
                "pi05_boundary04",
                "pi05_classifier3",
                "pi05_sequential3",
                "gr00t_init",
                "gr00t_baseline",
                "gr00t_baseline_failure",
                "gr00t_echo07",
                "gr00t_boundary04",
                "gr00t_signal_isolated",
            ],
        )
        listed = {
            "experiments.json",
            "data.json",
            "signals.json",
            "gr00t_statistics.json",
        }
        for row in registry["experiments"]:
            init = row["key"].endswith("_init")
            path = ROOT / row["training_config"]
            cfg = (
                validate_config(read_json(path))
                if row["backbone"] == "gr00t"
                else load_training(path)
            )
            self.assertEqual(cfg["max_steps"], 10000 if init else 2500)
            self.assertEqual(cfg["world_size"], 8)
            listed.add(path.name)
            if init:
                self.assertNotIn("evaluation_config", row)
                continue
            evaluation = ROOT / row["evaluation_config"]
            listed.add(evaluation.name)
            self.assertEqual(read_json(evaluation)["max_steps"], 520)
            self.assertEqual(
                Path(read_json(evaluation)["checkpoint"]).parts[1], cfg["run_dir"][5:]
            )
        # Every configuration file belongs to a listed experiment.
        self.assertEqual({p.name for p in (ROOT / "configs").iterdir()}, listed)

    def test_signal_isolated_random_stream_leaves_global_draws_unchanged(self):
        cpu = torch.device("cpu")
        stream = native_model_module().RandomStream()
        with self.assertRaisesRegex(ValueError, "not seeded"):
            with stream.active(cpu):
                pass
        torch.manual_seed(7)
        expected = torch.randn(6)
        torch.manual_seed(7)
        stream.seed(7, cpu)  # seeding must not disturb the global stream
        first = torch.randn(3)
        with stream.active(cpu):
            inside = torch.randn(3)
        with stream.active(cpu):
            again = torch.randn(3)
        second = torch.randn(3)
        torch.testing.assert_close(torch.cat((first, second)), expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(inside, first))
        self.assertFalse(torch.equal(inside, again))  # the private stream advances

    def test_signal_isolated_stream_resumes_exactly(self):
        stream = native_model_module().RandomStream()
        stream.seed(11, "cpu")
        with stream.active(torch.device("cpu")):
            torch.randn(5)
        saved = dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=None,
            status=stream.state,
        )
        with stream.active(torch.device("cpu")):
            expected = torch.randn(5)
        resumed = native_model_module().RandomStream()
        resumed.seed(99, "cpu")
        restore_rng(saved, resumed)
        with resumed.active(torch.device("cpu")):
            torch.testing.assert_close(torch.randn(5), expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "status random stream"):
            restore_rng(saved)

    def test_signal_isolated_masks_and_denominators(self):
        mask = native_model_module().status_attention_allowed(4, 3)
        self.assertEqual(mask.shape, (13, 13))
        self.assertTrue(mask[:7, :7].all())
        self.assertFalse(mask[:7, 7:].any())  # native tokens never read the signals
        self.assertTrue(mask[7:10, 7:10].all() and mask[10:, 10:].all())
        self.assertFalse(mask[7:10, 10:].any() or mask[10:, 7:10].any())
        batch = dict(
            action_mask=torch.ones(2, 16, 32, dtype=torch.bool),
            ss_mask=torch.ones(2, 16, dtype=torch.bool),
            fs_mask=torch.zeros(2, 16, dtype=torch.bool),
        )
        self.assertEqual(counts(batch, "isolated").tolist(), [1024, 32])
        self.assertEqual(counts(batch, "echo").tolist(), [1056])
        self.assertEqual(counts(batch, "baseline_failure").tolist(), [1024])

    def test_evaluation_determinism_before_cuda(self):
        # Interop thread settings can only be changed before parallel work;
        # exercise the actual entry setup in a fresh interpreter.
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                """
import os
import torch
from echo.evaluation.evaluate import configure_determinism
configure_determinism()
assert not torch.cuda.is_initialized()
assert torch.are_deterministic_algorithms_enabled()
assert torch.backends.cudnn.deterministic
assert not torch.backends.cudnn.benchmark
assert not torch.backends.cuda.matmul.allow_tf32
assert not torch.backends.cudnn.allow_tf32
assert torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
""",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
