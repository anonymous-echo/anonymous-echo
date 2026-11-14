"""Native GR00T inference (16-step action chunks) behind the common rollout interface."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from echo.models.gr00t.checkpoint import verify
from echo.models.gr00t.data import NativePreprocessor, native_collator, unnormalize_actions
from echo.models.gr00t.runtime import load_model, setup
from echo.io import read_json


class Policy:
    action_horizon = 16

    def __init__(
        self, checkpoint, tokenizer=None, device="cuda:0", num_inference_steps=8
    ):
        if num_inference_steps != 8:
            raise ValueError("GR00T inference uses eight steps")
        checkpoint = Path(checkpoint)
        receipt = verify(checkpoint)
        config = read_json(checkpoint / "training.json")["identity"]["config"]
        setup(config)
        self.statistics = read_json(checkpoint / "statistics.json")
        self.preprocess = NativePreprocessor(self.statistics, training=False)
        self.collate = native_collator()
        self.device = torch.device(device)
        self.model = load_model(config, self.device, checkpoint).eval()
        self.method = (
            config["method"] if config["method"] in {"echo", "isolated"} else "baseline"
        )
        self.last_signals = None
        self.metadata = dict(
            backbone="GR00T-N1.5",
            method=self.method,
            action_horizon=16,
            checkpoint_sha256=receipt["files"]["model.safetensors"]["sha256"],
            training_step=receipt["step"],
            statistics_sha256=receipt["files"]["statistics.json"]["sha256"],
            num_inference_steps=8,
            inference_precision="frozen_bf16_trainable_fp32_autocast_bf16",
        )

    @torch.inference_mode()
    def infer(self, observation, instruction):
        from echo.evaluation.evaluate import state_vector

        images = np.stack(
            [
                np.ascontiguousarray(observation[k][::-1, ::-1])
                for k in ("agentview_image", "robot0_eye_in_hand_image")
            ]
        )
        return self.infer_preprocessed(images, state_vector(observation), instruction)

    @torch.inference_mode()
    def infer_preprocessed(self, images, state, instruction):
        if (
            images.ndim != 4
            or images.shape[0] != 2
            or images.shape[-1] != 3
            or images.dtype != np.uint8
        ):
            raise ValueError("Expected two uint8 RGB camera images")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 8D observation state")
        sample = dict(
            image=[Image.fromarray(x) for x in images], state=state, lang=instruction
        )
        batch = self.collate([self.preprocess(sample)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.model.get_action(batch, num_steps=8)
        actions = unnormalize_actions(
            result.action_pred[0].float().cpu().numpy(), self.statistics["actions"]
        )
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid native GR00T action")
        actions[:, 6] = np.sign(actions[:, 6])
        self.last_signals = (
            {k: result[k][0].float().cpu().numpy() for k in ("ss", "fs")}
            if self.method in {"echo", "isolated"}
            else None
        )
        return actions

    def seed(self, seed):
        """Per-trial status stream; the global (action) stream is seeded by the caller."""
        if self.method == "isolated":
            self.model.status_stream.seed(seed, self.device)
