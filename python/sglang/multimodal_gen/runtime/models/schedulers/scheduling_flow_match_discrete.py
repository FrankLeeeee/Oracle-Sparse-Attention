# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/shengshu-ai/minWM
# (HY15/hyvideo/schedulers/scheduling_flow_match_discrete.py), itself modified
# from diffusers' FlowMatchEulerDiscreteScheduler.
"""Discrete flow-matching scheduler used by HunyuanVideo 1.5 / minWM.

Sigmas are ``linspace(1, 0, steps + 1)`` warped by the SD3 shift
``sigma' = shift * sigma / (1 + (shift - 1) * sigma)``; timesteps are
``sigma * num_train_timesteps``.  The Euler step is
``x_{i+1} = x_i + v * (sigma_{i+1} - sigma_i)`` (v-prediction).
"""

import torch


class FlowMatchDiscreteScheduler:
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

        # Fixed 1000-entry training sigma table for add_noise / x0 recovery.
        train_sigmas = torch.linspace(1, 0, num_train_timesteps + 1)[:-1]
        if shift != 1.0:
            train_sigmas = self.sd3_time_shift(train_sigmas)
        self.train_sigmas = train_sigmas

    def sd3_time_shift(self, t: torch.Tensor) -> torch.Tensor:
        return (self.shift * t) / (1 + (self.shift - 1) * t)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.num_inference_steps = num_inference_steps
        sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        sigmas = self.sd3_time_shift(sigmas)
        self.sigmas = sigmas.to(device=device)
        self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).to(
            device=device, dtype=torch.float32
        )

    def step_at(
        self,
        model_output: torch.Tensor,
        step_index: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler step from ``sigmas[step_index]`` to ``sigmas[step_index + 1]``."""
        dt = self.sigmas[step_index + 1] - self.sigmas[step_index]
        return sample.float() + model_output.float() * dt

    def _train_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        self.train_sigmas = self.train_sigmas.to(timestep.device)
        indices = timestep.long().clamp(0, len(self.train_sigmas) - 1)
        return self.train_sigmas[indices]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching: ``x_t = (1 - sigma) * x_0 + sigma * noise``."""
        sigma = self._train_sigma(timestep)
        while sigma.dim() < original_samples.dim():
            sigma = sigma.unsqueeze(-1)
        return ((1 - sigma) * original_samples + sigma * noise).type_as(
            original_samples
        )

    def pred_noise_to_pred_video(
        self,
        pred_noise: torch.Tensor,
        noise_input_latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """``x_0 = x_t - sigma * v`` for v-prediction flow matching."""
        sigma = self._train_sigma(timestep)
        while sigma.dim() < pred_noise.dim():
            sigma = sigma.unsqueeze(-1)
        return noise_input_latent - sigma * pred_noise
