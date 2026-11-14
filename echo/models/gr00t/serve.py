"""Native GR00T inference worker for the isolated GR00T Python environment."""

import argparse
import json
import os
import socket
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    if not args.socket.startswith("@") or len(args.socket) > 100:
        parser.error("Expected a short Linux abstract socket name")
    # The inference worker must die even if its evaluator is killed abruptly.
    # Install before importing CUDA/model packages; check the fork/exec race.
    from echo.models.gr00t.lifecycle import bind_to_parent

    bind_to_parent(args.parent_pid)
    from echo.models.gr00t.ipc import receive, send

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from echo.runtime import verify_runtime

    verified = verify_runtime("native")
    import torch
    from echo.models.gr00t.policy import Policy
    from echo.models.gr00t.runtime import seed_all

    torch.set_num_threads(2)
    seed_all(42)
    policy = Policy(
        args.checkpoint,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
    )
    from importlib.metadata import version

    runtime = dict(
        verified=verified,
        versions={
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "transformers",
                "Pillow",
                "imageio",
            )
        },
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(chr(0) + args.socket[1:])
        server.listen(1)
        connection, _ = server.accept()
        with connection:
            if hasattr(socket, "SO_PEERCRED"):
                _, uid, _ = struct.unpack(
                    "3i",
                    connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
                )
                if uid != os.getuid():
                    raise PermissionError("Inference client UID differs")
            send(
                connection,
                metadata=json.dumps(policy.metadata),
                runtime=json.dumps(runtime),
            )
            while True:
                request = receive(connection)
                kind = str(request["kind"])
                if kind == "stop":
                    break
                if kind == "seed":
                    seed_all(int(request["seed"]))
                    policy.seed(int(request["seed"]))
                    send(connection, ok=True)
                    continue
                if kind != "infer":
                    raise ValueError("Unknown inference request")
                actions = policy.infer_preprocessed(
                    request["images"], request["state"], str(request["instruction"])
                )
                response = dict(action=actions)
                if policy.last_signals is not None:
                    response.update(policy.last_signals)
                send(connection, **response)


if __name__ == "__main__":
    main()
