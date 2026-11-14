"""One categorical current-observation state; optional differentiable action conditioning."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from echo.models.model import (
    EchoPI05Adapter,
    EchoModelConfig,
    _attention_bias,
    _time_embedding,
)


def current_status_loss(
    action_error, action_mask, action_bound, logits, labels, weight
):
    """One CE term, independent of H50; explicit action+classification reduction."""
    action_sum = (action_error * action_mask * action_bound[..., None]).sum()
    action_count = action_mask.sum()
    action_loss = action_sum / action_count.clamp_min(1)
    ce = F.cross_entropy(logits, labels, reduction="mean")
    status_loss = ce * weight
    loss = action_loss + status_loss
    # The inherited trainer records ss/fs slots. They are explicitly relabeled
    # in metrics: ss is this ONE categorical objective; fs is absent (zero).
    count = logits.new_tensor(logits.shape[0], dtype=torch.long)
    zero = logits.sum() * 0
    return dict(
        loss=loss,
        action_loss=action_loss,
        ss_loss=status_loss,
        fs_loss=zero,
        action_numerator=action_loss * count,
        ss_numerator=status_loss * count,
        fs_numerator=zero,
        action_count=count,
        ss_count=count,
        fs_count=count * 0,
        loss_numerator=loss * count,
        loss_count=count,
        status_loss=ce,
        physical_action_entries=action_count,
        physical_action_numerator=action_sum,
    )


class StatusHeadPI05(EchoPI05Adapter):
    def __init__(
        self,
        base_model,
        config=None,
        method="classifier",
        status_classes=3,
        status_loss_weight=1.0,
    ):
        # Reuse Echo's batch/mask validation and prediction interface, without
        # creating any Echo signal projections or signal denoising tokens.
        nn.Module.__init__(self)
        if method not in {"classifier", "sequential"}:
            raise ValueError("Expected classifier or sequential")
        self.base = base_model
        self.echo_config = config or EchoModelConfig()
        self.method = method
        if status_classes != 3:
            raise ValueError("Expected exactly three mutually exclusive states")
        self.status_classes = status_classes
        self.status_loss_weight = float(status_loss_weight)
        if base_model.action_horizon != self.echo_config.horizon:
            raise ValueError("PI05 and signal horizon differ")
        width = base_model.vlm.config.text_config.hidden_size
        self.status_head = nn.Sequential(
            nn.Linear(width, 256), nn.SiLU(), nn.Linear(256, self.status_classes)
        )
        self.status_head.to(base_model.action_head.action_in_proj.weight)
        self.prefix_checkpointing = True

    def to_precision(self, precision):
        self.base.to_precision(precision)
        self.status_head.to(dtype=self.action_head.action_in_proj.weight.dtype)
        return self

    def _status_logits(self, hidden, mask):
        pooled = (hidden.float() * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[
            :, None
        ]
        return self.status_head(pooled.to(self.status_head[0].weight.dtype)).float()

    def _training_prefix_hidden(self, prefix, mask):
        """Checkpoint the native prefix layers; no detached cache in training.

        The pinned shared-layer implementation accepts a list of streams. We
        pass only the prefix stream, using its exact attention and normalization
        without a second transformer implementation. Actions recompute prefix.
        """
        from starVLA.model.framework.VLM4A.PI0 import _norm

        models = [self.base.vlm.language_model]
        bias = _attention_bias(mask[:, :, None] & mask[:, None, :])
        positions = mask.cumsum(1) - 1
        hidden = prefix
        for index in range(len(models[0].layers)):

            def layer(value, layer_index=index):
                return self.base.forward_shared_gemma_layer(
                    layer_index, models, [value], [None], bias, positions
                )[0]

            hidden = (
                checkpoint(layer, hidden, use_reentrant=False)
                if self.prefix_checkpointing and torch.is_grad_enabled()
                else layer(hidden)
            )
        return _norm(models[0].norm, hidden, None)[0]

    def _action_suffix(self, actions, probabilities, time):
        extra = actions.new_zeros((*actions.shape[:2], self.action_dim - 7))
        if self.method == "sequential":
            current = (2 * probabilities - 1)[:, None, :].expand(
                -1, actions.shape[1], -1
            )
            extra = torch.cat((current, extra[..., self.status_classes :]), dim=-1)
        padded = torch.cat((actions, extra), dim=-1)
        head = self.action_head
        suffix = head.action_in_proj(padded.to(head.action_in_proj.weight.dtype))
        t = _time_embedding(time, head.action_in_proj.out_features).to(
            head.time_mlp_in.weight.dtype
        )
        cond = F.silu(head.time_mlp_out(F.silu(head.time_mlp_in(t))))
        return suffix, cond

    def _action_velocity(self, prefix, mask, noisy, probabilities, time, cache=None):
        suffix, cond = self._action_suffix(noisy, probabilities, time)
        if (
            self.base.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix = suffix.bfloat16()
        batch, length = mask.shape
        horizon = self.action_horizon
        valid = torch.cat((mask, mask.new_ones((batch, horizon))), 1)
        allowed = valid[:, :, None] & valid[:, None, :]
        allowed[:, :length, length:] = False
        if cache is None:
            hidden = self.base.forward_with_action_expert(
                prefix_embs=prefix,
                suffix_embs=suffix,
                attention_mask=_attention_bias(allowed),
                position_ids=valid.cumsum(1) - 1,
                adarms_cond=cond,
            )
        else:
            self.action_head.set_gemma_expert_eager_attention()
            result = self.action_head.forward_gemma_expert(
                inputs_embeds=suffix,
                attention_mask=_attention_bias(allowed[:, length:]),
                position_ids=mask.sum(1)[:, None]
                + torch.arange(horizon, device=mask.device)[None],
                past_key_values=cache,
                use_cache=False,
                adarms_cond=cond,
            )
            hidden = result.last_hidden_state
        return self.action_head.action_out_proj(
            hidden.to(self.action_head.action_out_proj.weight.dtype)
        )[..., :7].float()

    def forward(self, examples, *, noise=None, time=None, return_debug=False):
        if not examples:
            raise ValueError("Empty status batch")
        batch = self.base._prepare_examples(examples, include_actions=True)
        observation = self.base._build_observation_from_batch(batch)
        device = self.action_head.action_in_proj.weight.device
        labels = [sample["status_label"] for sample in examples]
        if any(
            type(label) is not int or not 0 <= label < self.status_classes
            for label in labels
        ):
            raise ValueError("Expected one integer current-state label per observation")
        masks = {}
        for key, shape in (("echo_action_mask", (50, 7)), ("echo_action_bound", (50,))):
            value = np.stack([sample[key] for sample in examples])
            if value.dtype != np.bool_ or value.shape != (len(examples), *shape):
                raise ValueError("Invalid action mask/bound")
            masks[key] = torch.as_tensor(value, device=device)
        return self.forward_from_observation(
            observation,
            batch["action"].to(device=device, dtype=torch.float32)[..., :7],
            torch.tensor(labels, device=device, dtype=torch.long),
            action_mask=masks["echo_action_mask"],
            action_bound=masks["echo_action_bound"],
            noise=noise,
            time=time,
            return_debug=return_debug,
        )

    def forward_from_observation(
        self,
        observation,
        actions,
        labels,
        *,
        action_mask,
        action_bound,
        noise=None,
        time=None,
        return_debug=False,
    ):
        batch = actions.shape[0]
        if actions.shape != (batch, self.action_horizon, 7):
            raise ValueError("Expected physical H50 actions")
        noise = (
            self.action_head.sample_action_noise(
                (batch, self.action_horizon, 9), actions.device
            )
            if noise is None
            else torch.as_tensor(noise, device=actions.device, dtype=torch.float32)
        )
        if (
            noise.shape != (batch, self.action_horizon, 9)
            or not torch.isfinite(noise).all()
        ):
            raise ValueError("Expected finite [B,H,9] comparison noise")
        time = (
            self.action_head.sample_flow_time(batch, actions.device)
            if time is None
            else torch.as_tensor(time, device=actions.device, dtype=torch.float32)
        )
        if (
            time.shape != (batch,)
            or not torch.isfinite(time).all()
            or ((time < 0) | (time > 1)).any()
        ):
            raise ValueError("Expected finite diffusion time in [0,1]")
        prefix, mask, _ = self._prefix(observation)
        if (
            self.base.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix = prefix.bfloat16()
        logits = self._status_logits(self._training_prefix_hidden(prefix, mask), mask)
        probabilities = logits.softmax(dim=-1)
        noisy = (
            time[:, None, None] * noise[..., :7] + (1 - time[:, None, None]) * actions
        )
        velocity = self._action_velocity(prefix, mask, noisy, probabilities, time)
        action_error = (velocity - (noise[..., :7] - actions)).square()
        result = current_status_loss(
            action_error,
            action_mask,
            action_bound,
            logits,
            labels,
            self.status_loss_weight,
        )
        if return_debug:
            result.update(
                predicted_velocity=velocity,
                status_probabilities=probabilities,
                status_logits=logits,
                time=time,
            )
        return result

    @torch.inference_mode()
    def sample_actions(self, observation, noise=None, num_steps=10):
        if type(num_steps) is not int or num_steps < 1:
            raise ValueError("num_steps must be positive")
        prefix, mask, _ = self._prefix(observation)
        self.base.vlm.language_model.config._attn_implementation = "eager"
        output = self.base.vlm.forward_prefix(
            inputs_embeds=prefix,
            attention_mask=_attention_bias(mask[:, :, None] & mask[:, None, :]),
            position_ids=mask.cumsum(1) - 1,
            past_key_values=None,
            use_cache=True,
        )
        probabilities = self._status_logits(output.last_hidden_state, mask).softmax(
            dim=-1
        )
        shape = (prefix.shape[0], self.action_horizon, 9)
        x = (
            self.action_head.sample_action_noise(shape, prefix.device)
            if noise is None
            else torch.as_tensor(
                noise, device=prefix.device, dtype=torch.float32
            ).clone()
        )
        if x.shape != shape or not torch.isfinite(x).all():
            raise ValueError("Expected finite [B,H,9] comparison noise")
        x = x[..., :7]
        for step in range(num_steps):
            time = x.new_full((x.shape[0],), 1 - step / num_steps)
            x = (
                x
                - self._action_velocity(
                    prefix, mask, x, probabilities, time, cache=output.past_key_values
                )
                / num_steps
            )
        return dict(
            normalized_actions=F.pad(x, (0, self.action_dim - 7)),
            ss=probabilities[:, 1:2].expand(-1, self.action_horizon),
            fs=probabilities[:, 2:3].expand(-1, self.action_horizon),
            status_probabilities=probabilities,
            status_label=probabilities.argmax(dim=-1),
        )


def build_status_training_model(
    checkpoint,
    tokenizer,
    horizon=50,
    gradient_checkpointing=True,
    device="cuda",
    echo_config=None,
    method="classifier",
    status_classes=3,
    status_tail_frames=5,
    status_loss_weight=1.0,
):
    from dataclasses import asdict
    from omegaconf import OmegaConf
    from echo.models.initialization import build_training_model

    cfg = EchoModelConfig(**echo_config)
    base, metadata, stats = build_training_model(
        checkpoint,
        tokenizer,
        horizon=horizon,
        gradient_checkpointing=gradient_checkpointing,
        device=device,
    )
    model = StatusHeadPI05(
        base,
        cfg,
        method=method,
        status_classes=status_classes,
        status_loss_weight=status_loss_weight,
    )
    model.prefix_checkpointing = gradient_checkpointing
    OmegaConf.update(base.config, "echo", asdict(cfg), force_add=True)
    OmegaConf.update(base.config, "status_method", method, force_add=True)
    for name, value in (
        ("status_classes", status_classes),
        ("status_tail_frames", status_tail_frames),
        ("status_loss_weight", status_loss_weight),
    ):
        OmegaConf.update(base.config, name, value, force_add=True)
    metadata.update(
        variant=method,
        status_method=method,
        status_classes=status_classes,
        status_tail_frames=status_tail_frames,
        status_loss_weight=status_loss_weight,
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameter_count=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        known_unused_parameter_names=[
            "base." + n for n in metadata["known_unused_parameter_names"]
        ],
        architecturally_frozen_parameter_names=[
            "base." + n for n in metadata["architecturally_frozen_parameter_names"]
        ],
        signal_objective="one current-state cross-entropy per sample; mean reduction; coefficient status_loss_weight",
        action_objective="7D physical action masked flow mean; failure mask0; recovery bound excludes padding",
        status_inputs="pooled final observation prefix; no action features or GT status",
        action_conditioning="predicted current K-class probabilities, broadcast to action horizon; differentiable"
        if method == "sequential"
        else "none",
        training_prefix="checkpointed extra pass; inference reuses prefix KV cache",
    )
    return model, metadata, stats
