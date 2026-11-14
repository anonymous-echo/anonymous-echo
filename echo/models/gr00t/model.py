"""Echo on native N1.5: independent status tokens and asynchronous flow matching.

All original state/action/VLM weights retain their names and shapes. Time in
GR00T runs from noise (0) to clean (1); PI05's noise time is 1-t.

IsolatedSignalGR00T keeps the native action network untouched and predicts the
stop/failure signals with a separately parameterized copy, so signal training
cannot change the actions.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.feature_extraction_utils import BatchFeature


def signal_progress(action_progress, forcing):
    if not 0 <= forcing < 1:
        raise ValueError("forcing must be in [0, 1)")
    return (action_progress / (1 - forcing)).clamp(0, 1)


def attention_allowed(prefix, horizon, device=None):
    """Query rows/key columns; native state/future tokens retain action access.

    State/future queries cannot read SS/FS directly. Action queries see all;
    SS and FS see native tokens and their own block, never each other's block.
    As in PI05 mode5, indirect communication through action tokens is allowed.
    """
    native = prefix + horizon
    size = prefix + 3 * horizon
    mask = torch.zeros(size, size, dtype=torch.bool, device=device)
    mask[:native, :native] = True
    mask[prefix:native, :] = True
    mask[native:, :native] = True
    mask[native : native + horizon, native : native + horizon] = True
    mask[native + horizon :, native + horizon :] = True
    return mask


def token_norm(self, x, temb=None):
    scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
    if temb.ndim == 2:
        scale, shift = scale[:, None], shift[:, None]
    return self.norm(x) * (1 + scale) + shift


def dit_forward(model, hidden, visual, times, allowed=None, checkpoint_blocks=False):
    """Same native blocks/parameters, with optional tokenwise time and self mask."""
    temb = model.timestep_encoder(times.flatten()).reshape(*times.shape, -1)
    if times.ndim == 1:
        temb = temb[:, None].expand(-1, hidden.shape[1], -1)
    hidden, visual = hidden.contiguous(), visual.contiguous()
    bias = None
    if allowed is not None:
        bias = torch.zeros(allowed.shape, dtype=hidden.dtype, device=hidden.device)
        bias.masked_fill_(~allowed, float("-inf"))
        bias = bias[None].expand(hidden.shape[0], -1, -1)
    for i, block in enumerate(model.transformer_blocks):
        is_self = i % 2 == 1 and model.config.interleave_self_attention

        # Bind the layer in a default argument: checkpoint recomputes it later.
        def run(x, v, t, layer=block, self_attention=is_self):
            return layer(
                x,
                attention_mask=bias if self_attention else None,
                encoder_hidden_states=None if self_attention else v,
                temb=t,
            )

        if checkpoint_blocks and torch.is_grad_enabled():
            hidden = checkpoint(run, hidden, visual, temb, use_reentrant=False)
        else:
            hidden = run(hidden, visual, temb)
    shift, scale = model.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
    return model.proj_out_2(model.norm_out(hidden) * (1 + scale) + shift)


class SignalProjection(nn.Module):
    def __init__(self, width, output_width):
        super().__init__()
        self.input = nn.Linear(1, width)
        self.output = nn.Linear(output_width, 1)


def install_checkpointing(base):
    """Use the same block checkpointing in both branches without changing weights."""
    dit = base.action_head.model
    for block in dit.transformer_blocks:
        if block.norm_type == "ada_norm":
            block.norm1.forward = MethodType(token_norm, block.norm1)

    def run(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
    ):
        if return_all_hidden_states:
            raise ValueError("Intermediate states are not used in this experiment")
        return dit_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            checkpoint_blocks=self.training,
        )

    dit.forward = MethodType(run, dit)


class BaselineGR00T(nn.Module):
    def __init__(self, base, gradient_checkpointing=True):
        super().__init__()
        self.base = base
        if gradient_checkpointing:
            install_checkpointing(base)

    def forward(self, inputs):
        inputs = dict(inputs)
        mask = inputs["action_mask"]
        bounded = mask & inputs["action_bound"][..., None]
        count = mask.sum()
        # Native forward divides by mask.sum(). Keep a finite connected graph
        # on an all-failure microbatch, then multiply its contribution by zero.
        # This does not teach the sentinel target or alter any supervised batch.
        safe_mask = bounded
        if not bounded.any():
            safe_mask = bounded.clone()
            safe_mask.reshape(-1)[0] = True
        inputs["action_mask"] = safe_mask
        result = self.base(inputs)
        result["loss"] = (
            result["loss"]
            * bounded.sum().to(result["loss"].device)
            / count.clamp_min(1).to(result["loss"].device)
        )
        result["action_loss"] = result["loss"]
        result["action_count"] = count.to(result["loss"].device)
        result["loss_count"] = result["action_count"]
        result["loss_numerator"] = result["loss"] * result["loss_count"]
        return result

    def get_action(self, inputs, num_steps=8):
        self.base.action_head.num_inference_timesteps = num_steps
        return self.base.get_action(inputs)


class EchoGR00T(nn.Module):
    def __init__(self, base, forcing=0.7, gradient_checkpointing=True):
        super().__init__()
        self.base = base
        self.forcing = forcing
        signal_progress(torch.tensor(0.0), forcing)
        self.gradient_checkpointing = gradient_checkpointing
        head = base.action_head
        self.ss_projection = SignalProjection(
            head.input_embedding_dim, head.hidden_size
        )
        self.fs_projection = SignalProjection(
            head.input_embedding_dim, head.hidden_size
        )
        for module in (self.ss_projection, self.fs_projection):
            module.to(device=head.device, dtype=head.dtype)
        for block in head.model.transformer_blocks:
            if block.norm_type != "ada_norm":
                raise ValueError("Expected native adaptive layer normalization")
            block.norm1.forward = MethodType(token_norm, block.norm1)

    def velocity(self, backbone, inputs, actions, ss, fs, progress):
        head = self.base.action_head
        batch, horizon = actions.shape[:2]
        t = (progress * head.num_timestep_buckets).long()
        st = (
            signal_progress(progress, self.forcing) * head.num_timestep_buckets
        ).long()
        embodiment = inputs.embodiment_id
        state = head.state_encoder(inputs.state, embodiment)
        future = head.future_tokens.weight[None].expand(batch, -1, -1)
        action = head.action_encoder(actions, t, embodiment)
        ss_tokens, fs_tokens = (
            self.ss_projection.input(ss),
            self.fs_projection.input(fs),
        )
        # Time enters the native DiT's AdaLN on every token. Action encoder also
        # retains its native time features; signal inputs have independent weights.
        if head.config.add_pos_embed:
            pos = head.position_embedding(torch.arange(horizon, device=actions.device))[
                None
            ]
            action, ss_tokens, fs_tokens = (
                action + pos,
                ss_tokens + pos,
                fs_tokens + pos,
            )
        prefix = state.shape[1] + future.shape[1]
        tokens = torch.cat((state, future, action, ss_tokens, fs_tokens), 1)
        times = torch.cat(
            (
                t[:, None].expand(-1, prefix + horizon),
                st[:, None].expand(-1, 2 * horizon),
            ),
            1,
        )
        hidden = dit_forward(
            head.model,
            tokens,
            backbone.backbone_features,
            times,
            attention_allowed(prefix, horizon, actions.device),
            self.gradient_checkpointing,
        )
        a, s, f = hidden[:, prefix:].split(horizon, 1)
        return (
            head.action_decoder(a, embodiment),
            self.ss_projection.output(s),
            self.fs_projection.output(f),
        )

    def forward(self, inputs):
        backbone_inputs, inputs = self.base.prepare_input(inputs)
        head = self.base.action_head
        head.set_frozen_modules_to_eval_mode()
        backbone = head.process_backbone_output(self.base.backbone(backbone_inputs))
        actions = inputs.action
        # Signals use the same binary scaling as the PI05 Echo reference.
        signals = [inputs[k][..., None] / (1 + 1e-6) * 2 - 1 for k in ("ss", "fs")]
        targets = [actions, *signals]
        noise = [torch.randn_like(x) for x in targets]
        progress = head.sample_time(len(actions), actions.device, actions.dtype)
        sp = signal_progress(progress, self.forcing)
        noisy = [
            (1 - t[:, None, None]) * n + t[:, None, None] * x
            for x, n, t in zip(targets, noise, (progress, sp, sp))
        ]
        predictions = self.velocity(backbone, inputs, *noisy, progress)
        errors = [
            (p.float() - (x - n).float()).square()
            for p, x, n in zip(predictions, targets, noise)
        ]
        masks = [
            inputs.action_mask,
            inputs.ss_mask[..., None],
            inputs.fs_mask[..., None],
        ]
        # Keep recovery padding in the denominator but zero its action gradient,
        # as in the baseline (a separate bound, not a new mask).
        bounds = [inputs.action_bound[..., None], 1, 1]
        scale = torch.where(inputs.padding_mask.any(1), 1.0, 0.1)[:, None, None]
        errors[1] = errors[1] * scale
        numerators = [(e * m * b).sum() for e, m, b in zip(errors, masks, bounds)]
        counts = [m.sum() for m in masks]
        result = {
            "loss": sum(numerators) / sum(counts).clamp_min(1),
            "loss_numerator": sum(numerators),
            "loss_count": sum(counts),
        }
        for name, n, c in zip(("action", "ss", "fs"), numerators, counts):
            result[name + "_loss"] = n / c.clamp_min(1)
            result[name + "_count"] = c
        return BatchFeature(result)

    @torch.no_grad()
    def get_action(self, inputs, num_steps=8):
        backbone_inputs, inputs = self.base.prepare_input(inputs)
        head = self.base.action_head
        backbone = head.process_backbone_output(self.base.backbone(backbone_inputs))
        batch = inputs.state.shape[0]
        actions = torch.randn(
            batch,
            head.action_horizon,
            head.action_dim,
            device=head.device,
            dtype=backbone.backbone_features.dtype,
        )
        ss = torch.randn(
            batch, head.action_horizon, 1, device=head.device, dtype=head.dtype
        )
        fs = torch.randn_like(ss)
        for i in range(num_steps):
            t = torch.full(
                (batch,), i / num_steps, device=head.device, dtype=head.dtype
            )
            nxt = torch.full_like(t, (i + 1) / num_steps)
            va, vs, vf = self.velocity(backbone, inputs, actions, ss, fs, t)
            ds = (
                signal_progress(nxt, self.forcing) - signal_progress(t, self.forcing)
            )[:, None, None]
            actions = actions + (1 / num_steps) * va
            ss, fs = ss + ds * vs, fs + ds * vf
        return BatchFeature(
            data=dict(
                action_pred=actions, ss=(ss[..., 0] + 1) / 2, fs=(fs[..., 0] + 1) / 2
            )
        )


STATUS_SEED_OFFSET = 1_000_003


def status_attention_allowed(prefix, horizon, device=None):
    """Self-attention mask of the status network (query rows, key columns).

    Its state/future/action tokens see one another only. SS and FS see those
    tokens and their own block, never each other's block.
    """
    native = prefix + horizon
    size = prefix + 3 * horizon
    mask = torch.zeros(size, size, dtype=torch.bool, device=device)
    mask[:native, :native] = True
    mask[native:, :native] = True
    mask[native : native + horizon, native : native + horizon] = True
    mask[native + horizon :, native + horizon :] = True
    return mask


class RandomStream:
    """Private CPU/CUDA random state; the global streams are restored on exit.

    Status noise and time draws run here, so the action network sees exactly
    the random sequence it would see without the status network.
    """

    def __init__(self):
        self.state = None

    @staticmethod
    def _devices(device):
        return [device] if device.type == "cuda" else []

    @staticmethod
    def _capture(device):
        cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        return torch.get_rng_state(), cuda

    def seed(self, seed, device):
        device = torch.device(device)
        with torch.random.fork_rng(devices=self._devices(device)):
            torch.random.default_generator.manual_seed(seed + STATUS_SEED_OFFSET)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed + STATUS_SEED_OFFSET)
            self.state = self._capture(device)

    @contextmanager
    def active(self, device):
        device = torch.device(device)
        if self.state is None:
            raise ValueError("Status random stream is not seeded")
        with torch.random.fork_rng(devices=self._devices(device)):
            torch.set_rng_state(self.state[0])
            if device.type == "cuda":
                torch.cuda.set_rng_state(self.state[1], device)
            yield
            self.state = self._capture(device)


class StatusDiT(nn.Module):
    """Independent copy of every trainable action-head module, predicting SS/FS.

    Values are copied from the action head after the checkpoint loads. The only
    shared input is the frozen VLM output, which is read detached.
    """

    COPIED = (
        "vlln",
        "vl_self_attention",
        "state_encoder",
        "action_encoder",
        "future_tokens",
        "model",
    )

    def __init__(self, head, forcing, gradient_checkpointing=True):
        super().__init__()
        signal_progress(torch.tensor(0.0), forcing)
        self.forcing = forcing
        self.gradient_checkpointing = gradient_checkpointing
        self.add_pos_embed = head.config.add_pos_embed
        self.num_timestep_buckets = head.num_timestep_buckets
        self.action_horizon = head.action_horizon
        for name in self.names():
            setattr(self, name, copy.deepcopy(getattr(head, name)))
        self.ss_projection = SignalProjection(
            head.input_embedding_dim, head.hidden_size
        )
        self.fs_projection = SignalProjection(
            head.input_embedding_dim, head.hidden_size
        )
        for module in (self.ss_projection, self.fs_projection):
            module.to(device=head.device, dtype=head.dtype)
        for block in self.model.transformer_blocks:
            if block.norm_type != "ada_norm":
                raise ValueError("Expected native adaptive layer normalization")
            block.norm1.forward = MethodType(token_norm, block.norm1)

    def names(self):
        return self.COPIED + (("position_embedding",) if self.add_pos_embed else ())

    def copy_from(self, head):
        for name in self.names():
            getattr(self, name).load_state_dict(getattr(head, name).state_dict())

    def visual(self, raw):
        return self.vl_self_attention(self.vlln(raw.detach()))

    def velocity(self, visual, inputs, actions, ss, fs, progress):
        batch, horizon = actions.shape[:2]
        t = (progress * self.num_timestep_buckets).long()
        st = (
            signal_progress(progress, self.forcing) * self.num_timestep_buckets
        ).long()
        embodiment = inputs.embodiment_id
        state = self.state_encoder(inputs.state, embodiment)
        future = self.future_tokens.weight[None].expand(batch, -1, -1)
        action = self.action_encoder(actions, t, embodiment)
        ss_tokens, fs_tokens = (
            self.ss_projection.input(ss),
            self.fs_projection.input(fs),
        )
        if self.add_pos_embed:
            pos = self.position_embedding(torch.arange(horizon, device=actions.device))[
                None
            ]
            action, ss_tokens, fs_tokens = (
                action + pos,
                ss_tokens + pos,
                fs_tokens + pos,
            )
        prefix = state.shape[1] + future.shape[1]
        tokens = torch.cat((state, future, action, ss_tokens, fs_tokens), 1)
        times = torch.cat(
            (
                t[:, None].expand(-1, prefix + horizon),
                st[:, None].expand(-1, 2 * horizon),
            ),
            1,
        )
        hidden = dit_forward(
            self.model,
            tokens,
            visual,
            times,
            status_attention_allowed(prefix, horizon, actions.device),
            self.gradient_checkpointing,
        )
        _, s, f = hidden[:, prefix:].split(horizon, 1)
        return self.ss_projection.output(s), self.fs_projection.output(f)

    def losses(self, raw, inputs, sample_time):
        """SS/FS flow-matching numerators and counts."""
        visual = self.visual(raw)
        actions = inputs.action
        signals = [inputs[k][..., None] / (1 + 1e-6) * 2 - 1 for k in ("ss", "fs")]
        noise = [torch.randn_like(x) for x in (actions, *signals)]
        progress = sample_time(len(actions), actions.device, actions.dtype)
        sp = signal_progress(progress, self.forcing)
        noisy = [
            (1 - t[:, None, None]) * n + t[:, None, None] * x
            for x, n, t in zip((actions, *signals), noise, (progress, sp, sp))
        ]
        predictions = self.velocity(visual, inputs, *noisy, progress)
        errors = [
            (p.float() - (x - n).float()).square()
            for p, x, n in zip(predictions, signals, noise[1:])
        ]
        scale = torch.where(inputs.padding_mask.any(1), 1.0, 0.1)[:, None, None]
        errors[0] = errors[0] * scale
        masks = [inputs.ss_mask[..., None], inputs.fs_mask[..., None]]
        return [(e * m).sum() for e, m in zip(errors, masks)], [m.sum() for m in masks]

    def denoise(self, raw, inputs, actions, num_steps):
        """SS/FS beside the executed chunk, conditioned on its predicted actions."""
        visual = self.visual(raw)
        clean = actions.to(self.ss_projection.input.weight.dtype)
        batch = clean.shape[0]
        noise = torch.randn_like(clean)
        ss = torch.randn(
            batch, self.action_horizon, 1, device=clean.device, dtype=clean.dtype
        )
        fs = torch.randn_like(ss)
        for i in range(num_steps):
            t = torch.full(
                (batch,), i / num_steps, device=clean.device, dtype=clean.dtype
            )
            nxt = torch.full_like(t, (i + 1) / num_steps)
            noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * clean
            vs, vf = self.velocity(visual, inputs, noisy, ss, fs, t)
            ds = (
                signal_progress(nxt, self.forcing) - signal_progress(t, self.forcing)
            )[:, None, None]
            ss, fs = ss + ds * vs, fs + ds * vf
        return ss, fs


class IsolatedSignalGR00T(nn.Module):
    """Native action network beside a separately parameterized status network.

    Actions follow the baseline-with-failure-data recipe exactly: native
    forward and get_action, failure chunks sampled with zero action loss, and
    the same all-failure sentinel. The status network shares only the frozen
    VLM output, read detached, and draws its randomness from its own stream.
    The training loop normalizes and clips the two parameter sets separately.
    """

    def __init__(self, base, forcing=0.7, gradient_checkpointing=True):
        super().__init__()
        self.base = base
        # Copy before checkpointing patches the action head's blocks.
        self.status = StatusDiT(base.action_head, forcing, gradient_checkpointing)
        if gradient_checkpointing:
            install_checkpointing(base)
        self.status_stream = RandomStream()

    def forward(self, inputs):
        inputs = dict(inputs)
        mask = inputs["action_mask"]
        bounded = mask & inputs["action_bound"][..., None]
        count = mask.sum()
        # Native forward divides by mask.sum(). Keep a finite connected graph
        # on an all-failure microbatch, then multiply its contribution by zero.
        safe_mask = bounded
        if not bounded.any():
            safe_mask = bounded.clone()
            safe_mask.reshape(-1)[0] = True
        inputs["action_mask"] = safe_mask
        # GR00T_N1_5.forward, keeping the frozen VLM output for the status network.
        backbone_inputs, action_inputs = self.base.prepare_input(inputs)
        backbone = self.base.backbone(backbone_inputs)
        raw = backbone["backbone_features"]
        result = self.base.action_head(backbone, action_inputs)
        self.base.validate_data(result, backbone, is_training=True)
        device = result["loss"].device
        result["loss"] = (
            result["loss"] * bounded.sum().to(device) / count.clamp_min(1).to(device)
        )
        result["action_loss"] = result["loss"]
        result["action_count"] = count.to(device)
        result["action_numerator"] = result["loss"] * result["action_count"]
        with self.status_stream.active(raw.device):
            (ss, fs), (ss_count, fs_count) = self.status.losses(
                raw, action_inputs, self.base.action_head.sample_time
            )
        result["status_numerator"] = ss + fs
        result["status_count"] = ss_count + fs_count
        for name, numerator, n in (("ss", ss, ss_count), ("fs", fs, fs_count)):
            result[name + "_loss"] = numerator / n.clamp_min(1)
            result[name + "_count"] = n
        result["loss_numerator"] = (
            result["action_numerator"] + result["status_numerator"]
        )
        result["loss_count"] = result["action_count"] + result["status_count"]
        return result

    @torch.no_grad()
    def get_action(self, inputs, num_steps=8):
        self.base.action_head.num_inference_timesteps = num_steps
        backbone_inputs, action_inputs = self.base.prepare_input(inputs)
        backbone = self.base.backbone(backbone_inputs)
        raw = backbone["backbone_features"]
        result = self.base.action_head.get_action(backbone, action_inputs)
        self.base.validate_data(result, backbone, is_training=False)
        with self.status_stream.active(raw.device):
            ss, fs = self.status.denoise(
                raw, action_inputs, result["action_pred"], num_steps
            )
        result["ss"], result["fs"] = (ss[..., 0] + 1) / 2, (fs[..., 0] + 1) / 2
        return result
