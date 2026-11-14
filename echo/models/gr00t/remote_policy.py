"""Simulator-side adapter for the native GR00T inference process."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

import numpy as np

from echo.models.gr00t.ipc import receive, send


class RemotePolicy:
    action_horizon = 16

    def __init__(
        self,
        checkpoint,
        tokenizer,
        device,
        num_inference_steps,
        *,
        python,
        run_dir,
        shard_index,
    ):
        address = "@echo-gr00t-" + uuid.uuid4().hex
        runtime = Path(run_dir) / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        self.log = (runtime / f"native_shard_{shard_index:03d}.log").open("a")
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PYTHONPATH"] = os.pathsep.join(
            (str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", ""))
        )
        try:
            self.process = subprocess.Popen(
                [
                    str(Path(python).absolute()),
                    "-B",
                    "-u",
                    "-m",
                    "echo.models.gr00t.serve",
                    "--socket",
                    address,
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    str(device),
                    "--num-inference-steps",
                    str(num_inference_steps),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            self.log.close()
            raise
        self.connection = None
        self.last_signals = None
        try:
            self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.connection.settimeout(600)
            deadline = time.monotonic() + 1200
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Native inference exited; inspect {self.log.name}"
                    )
                try:
                    self.connection.connect(chr(0) + address[1:])
                    break
                except ConnectionRefusedError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Native inference startup")
                    time.sleep(1)
            ready = receive(self.connection)
            self.metadata = json.loads(str(ready["metadata"]))
            self.native_runtime = json.loads(str(ready["runtime"]))
            self.method = self.metadata["method"]
        except BaseException:
            self.close()
            raise

    def seed(self, value):
        send(self.connection, kind="seed", seed=value)
        if not bool(receive(self.connection)["ok"]):
            raise ValueError("Native seed was not acknowledged")

    def infer(self, observation, instruction):
        from echo.evaluation.evaluate import state_vector

        images = np.stack(
            [
                np.ascontiguousarray(observation[k][::-1, ::-1])
                for k in ("agentview_image", "robot0_eye_in_hand_image")
            ]
        )
        send(
            self.connection,
            kind="infer",
            images=images,
            state=state_vector(observation),
            instruction=instruction,
        )
        value = receive(self.connection)
        actions = value["action"]
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid native action response")
        self.last_signals = (
            {name: value[name] for name in ("ss", "fs")} if "ss" in value else None
        )
        return actions

    def close(self):
        connection = getattr(self, "connection", None)
        process = getattr(self, "process", None)
        if connection is not None:
            try:
                connection.settimeout(1)
                if process is not None and process.poll() is None:
                    send(connection, kind="stop")
            except (OSError, ValueError):
                pass
            finally:
                connection.close()
                self.connection = None
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait()
            self.process = None
        log = getattr(self, "log", None)
        if log is not None:
            log.close()
            self.log = None
