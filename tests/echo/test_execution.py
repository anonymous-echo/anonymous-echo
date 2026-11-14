"""Cross-process array fidelity, cleanup and current-observation metric edges."""

import io
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from echo.evaluation.signals import categorical_accuracy
from echo.models.gr00t.ipc import MAX_PAYLOAD, receive, send
from echo.models.gr00t.remote_policy import RemotePolicy
from echo.runtime import verify_runtime


class ExecutionContracts(unittest.TestCase):
    def test_runtime_rejects_cross_environment_packages(self):
        from importlib.metadata import PackageNotFoundError

        root = Path(__file__).resolve().parents[2]
        expected = json.loads(
            (root / "assets/runtime/environments.json").read_text()
        )
        with (
            patch("echo.runtime.platform.system", return_value="Linux"),
            patch("echo.runtime.platform.python_version", return_value="3.10.18"),
        ):
            for profile in ("native", "simulator"):
                packages = expected[profile]["packages"]
                with patch("echo.runtime.version", side_effect=packages.__getitem__):
                    self.assertEqual(verify_runtime(profile)["packages"], packages)
                wrong = dict(packages, torch="incorrect")
                with patch("echo.runtime.version", side_effect=wrong.__getitem__):
                    with self.assertRaisesRegex(RuntimeError, "torch"):
                        verify_runtime(profile)
                with patch(
                    "echo.runtime.version", side_effect=PackageNotFoundError("missing")
                ):
                    with self.assertRaisesRegex(RuntimeError, "environment differs"):
                        verify_runtime(profile)

    def test_inference_wire_preserves_values_with_fragmented_reads(self):
        a, b = socket.socketpair()

        class Fragmented:
            def recv(self, size):
                return b.recv(min(size, 7))

        try:
            values = dict(
                kind="infer",
                instruction="open the drawer",
                seed=np.int64(42),
                images=np.arange(96, dtype=np.uint8).reshape(2, 4, 4, 3),
                state=np.linspace(-1, 1, 8, dtype=np.float32),
            )
            send(a, **values)
            output = receive(Fragmented())
            self.assertEqual(set(output), set(values))
            for key, value in values.items():
                np.testing.assert_array_equal(output[key], value)
                self.assertEqual(output[key].dtype, np.asarray(value).dtype)
        finally:
            a.close()
            b.close()

    def test_inference_wire_rejects_truncation_and_oversized_header(self):
        for payload, error in [
            (b"\x00\x00", EOFError),
            (struct.pack("!Q", MAX_PAYLOAD + 1), ValueError),
        ]:
            a, b = socket.socketpair()
            try:
                a.sendall(payload)
                a.close()
                with self.assertRaises(error):
                    receive(b)
            finally:
                a.close()
                b.close()

    def test_native_close_reaps_unresponsive_child_and_is_idempotent(self):
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                'import time;print("ready",flush=True);time.sleep(60)',
            ],
            stdout=subprocess.PIPE,
        )
        self.assertEqual(process.stdout.readline().strip(), b"ready")
        policy = RemotePolicy.__new__(RemotePolicy)
        policy.process, policy.connection, policy.log = process, None, io.StringIO()
        try:
            policy.close()
            self.assertIsNotNone(process.poll())
            policy.close()
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()

    def test_native_socket_setup_failure_still_cleans_up(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            with (
                patch("echo.models.gr00t.remote_policy.subprocess.Popen") as popen,
                patch(
                    "echo.models.gr00t.remote_policy.socket.socket",
                    side_effect=OSError("socket unavailable"),
                ),
            ):
                process = popen.return_value
                with self.assertRaisesRegex(OSError, "socket unavailable"):
                    RemotePolicy(
                        "checkpoint",
                        None,
                        "cuda:0",
                        8,
                        python=".venv-gr00t/bin/python",
                        run_dir=tmp,
                        shard_index=0,
                    )
                process.wait.assert_called()
                # Keep the virtualenv path instead of resolving its symlink.
                self.assertIn(".venv-gr00t/bin/python", popen.call_args.args[0][0])
                self.assertTrue(popen.call_args.kwargs["stdout"].closed)

    def test_categorical_failure_argmax_and_final_action_success(self):
        queries = np.arange(0, 520, 8)
        probabilities = np.tile([0.25, 0.35, 0.4], (len(queries), 1))
        events = np.zeros(520, dtype=bool)
        row = categorical_accuracy(events, queries, probabilities)
        self.assertEqual(row["failure_prediction_frames"], 520)
        self.assertEqual(row["frame_correct"], 0)
        self.assertFalse(row["terminal_failure_gt_available"])
        events[-1] = True
        row = categorical_accuracy(events, queries, probabilities)
        self.assertTrue(row["success"])
        self.assertEqual(row["success_frame_total"], 0)
        events[:] = False
        events[7] = True
        row = categorical_accuracy(events, queries, probabilities)
        self.assertEqual(row["ongoing_frame_total"], 8)
        self.assertEqual(row["success_frame_total"], 61)


if __name__ == "__main__":
    unittest.main()
