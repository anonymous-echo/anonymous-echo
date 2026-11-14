"""Raw portable chunks, canonical prompts, and native GR00T preprocessing.

Raw mode bypasses quantile normalization before native min/max normalization.
It never attempts to recover raw actions by inverting a normalizer.
"""

from __future__ import annotations

import numpy as np


def raw_dataset(manifest, data_config, method="echo", **options):
    from echo.data.chunks import ChunkDataset

    unused = {
        k: {"q01": [-1.0] * n, "q99": [1.0] * n}
        for k, n in [("actions", 7), ("state", 8)]
    }
    return ChunkDataset(
        manifest, data_config, unused, method=method, horizon=16, raw=True, **options
    )


def minmax(value, statistics):
    value = np.asarray(value, dtype=np.float32)
    low, high = (np.asarray(statistics[k], dtype=np.float32) for k in ("min", "max"))
    result = np.zeros_like(value)
    active = high != low
    result[..., active] = (
        2 * (value[..., active] - low[active]) / (high[active] - low[active]) - 1
    )
    return result


def unnormalize_actions(value, statistics):
    low, high = (np.asarray(statistics[k], dtype=np.float32) for k in ("min", "max"))
    return (np.asarray(value)[..., :7] + 1) * (high - low) / 2 + low


class NativePreprocessor:
    """Official LIBERO image transforms and Eagle chat processor; no language rewrite."""

    def __init__(self, statistics, training=True, signals=True):
        from gr00t.model.transforms import GR00TTransform
        from gr00t.data.embodiment_tags import EmbodimentTag

        self.statistics = statistics
        self.training = training
        self.signals = signals
        self.model_transform = GR00TTransform(
            state_horizon=1,
            action_horizon=16,
            max_state_dim=64,
            max_action_dim=32,
            training=training,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        )

    def __call__(self, sample):
        import torch
        import torchvision.transforms.v2 as T

        images = [np.asarray(x.convert("RGB")) for x in sample["image"]]
        if images[0].shape != images[1].shape:
            raise ValueError("The two camera resolutions must match")
        views = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
        h, w = views.shape[-2:]
        size = (int(h * 0.95), int(w * 0.95))
        views = (T.RandomCrop(size) if self.training else T.CenterCrop(size))(views)
        views = T.Resize(
            (224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True
        )(views)
        if self.training:
            views = T.ColorJitter(
                brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08
            )(views)
        video = (views.permute(0, 2, 3, 1) * 255).to(torch.uint8).numpy()[None]
        state = np.asarray(sample["state"], dtype=np.float32)
        if state.shape == (8,):
            state = state[None]
        if state.shape != (1, 8):
            raise ValueError(
                f"Expected one 8D LIBERO observation, received {state.shape}"
            )
        inputs = {
            "video": video,
            "state": minmax(state, self.statistics["state"]),
            "annotation.human.action.task_description": [sample["lang"]],
        }
        if self.training:
            inputs["action"] = minmax(sample["action"], self.statistics["actions"])
        result = self.model_transform(inputs)
        if self.training:
            result["action_mask"][:, :7] &= sample["echo_action_mask"]
            result["action_bound"] = sample["echo_action_bound"]
            result["padding_mask"] = sample["padding_mask"]
            for signal in ("ss", "fs") if self.signals else ():
                result[signal] = np.asarray(sample["echo_" + signal], dtype=np.float32)
                result[signal + "_mask"] = (
                    sample["echo_" + signal + "_mask"]
                    & sample["echo_" + signal + "_bound"]
                )
        return result


class NativeDataset:
    def __init__(
        self, manifest, data_config, statistics, method="echo", training=True, **options
    ):
        self._constructor = dict(
            manifest=manifest,
            data_config=data_config,
            statistics=statistics,
            method=method,
            training=training,
            **options,
        )
        self.reader = raw_dataset(manifest, data_config, method=method, **options)
        self.preprocess = NativePreprocessor(
            statistics, training, signals=method != "baseline_failure"
        )
        self.segments, self.offsets = self.reader.segments, self.reader.offsets

    def __getstate__(self):
        # Spawned workers rebuild the reader and never inherit open episodes.
        return self._constructor

    def __setstate__(self, state):
        self.__init__(**state)

    def __len__(self):
        return len(self.reader)

    def process(self, index):
        sample = self.reader[index]
        if self.reader.method == "baseline":
            from echo.data.chunks import signal_targets
            from bisect import bisect_right

            position = bisect_right(self.reader.offsets, index) - 1
            local = index - self.reader.offsets[position]
            length = self.reader.offsets[position + 1] - self.reader.offsets[position]
            pad, ss, sm, fs, fm, am, ab, sb, fb = signal_targets(
                sample["segment_type"], length, local, 16
            )
            sample.update(
                echo_action_mask=am,
                echo_action_bound=ab,
                padding_mask=pad,
                echo_ss=ss,
                echo_fs=fs,
                echo_ss_mask=sm,
                echo_fs_mask=fm,
                echo_ss_bound=sb,
                echo_fs_bound=fb,
            )
        result = self.preprocess(sample)
        result["audit_type"] = np.int64(
            {"success": 0, "recovery": 1, "failure": 2}[sample["segment_type"]]
        )
        result["audit_padding"] = np.bool_(sample["padding_mask"].any())
        return result

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            return self.process(index)
        # Per-draw augmentation makes worker prefetching irrelevant to resume.
        import random
        import torch

        frame, draw_seed = index
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                random.seed(draw_seed)
                np.random.seed(draw_seed % 2**32)
                torch.random.default_generator.manual_seed(draw_seed)
                return self.process(frame)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def native_collator():
    from gr00t.model.transforms import DefaultDataCollator

    return DefaultDataCollator()
