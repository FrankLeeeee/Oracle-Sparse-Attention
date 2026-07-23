# SPDX-License-Identifier: Apache-2.0
"""Pipeline config for the causal (minWM) HunyuanVideo 1.5 TI2V model.

Encoder stack (identical to stock HunyuanVideo 1.5):

* ``text_encoder``: Qwen2.5-VL-7B text backbone, hidden state at skip-2, chat
  template cropped at token 108 (verified against the checkpoint tokenizer).
* ``text_encoder_2``: ByT5-small with Glyph-SDXL-v2 weights (1472-dim).
  minWM's rollout script encodes the full caption (no glyph extraction).
* ``image_encoder``: SigLIP so400m; the 729 semantic tokens join the text
  stream inside the DiT.

The VAE-encoded first frame becomes a channel-concat conditioning latent
(built by the causal denoising stage), not a KV-cache warm-up input.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.hunyuanvideo15 import (
    CausalHunyuanVideo15Config,
)
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    SiglipVisionConfig,
    T5Config,
)
from sglang.multimodal_gen.configs.models.encoders.qwen_image import (
    Qwen2_5VLConfig,
    QwenImageArchConfig,
)
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig
from sglang.multimodal_gen.configs.models.vaes import HunyuanVideo15VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
    TextConditioningOutput,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)

# Rendered Qwen chat template (system + user + generation prompt).  Tokenizing
# this flat string is id-identical to ``tokenizer.apply_chat_template`` with
# minWM's ``li-dit-encode-video-json`` messages; the prompt content starts at
# token 108.
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|im_start|>system\nYou are a helpful assistant. Describe the video by "
    "detailing the following aspects:         1. The main content and theme "
    "of the video.         2. The color, shape, size, texture, quantity, "
    "text, and spatial relationships of the objects.         3. Actions, "
    "events, behaviors temporal relationships, physical movement changes of "
    "the objects.         4. background environment, light, style and "
    "atmosphere.         5. camera angles, movements, and transitions used "
    "in the video.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

PROMPT_CROP_START = 108
QWEN_MAX_LENGTH = 1000
BYT5_MAX_LENGTH = 256


def qwen_preprocess_text(prompt: str) -> str:
    return PROMPT_TEMPLATE_ENCODE_VIDEO.format(prompt)


def qwen_postprocess_text(
    outputs: BaseEncoderOutput, text_inputs
) -> TextConditioningOutput:
    hidden_state_skip_layer = 2
    assert outputs.hidden_states is not None
    last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
    last_hidden_state = last_hidden_state[:, PROMPT_CROP_START:]
    attention_mask = text_inputs.attention_mask.to(
        device=last_hidden_state.device, dtype=torch.bool
    )
    attention_mask = attention_mask[
        :, PROMPT_CROP_START : PROMPT_CROP_START + last_hidden_state.shape[1]
    ]
    seq_lens = [int(x) for x in attention_mask.to(torch.int64).sum(dim=1).tolist()]
    return TextConditioningOutput(last_hidden_state, attention_mask, seq_lens)


def byt5_postprocess_text(
    outputs: BaseEncoderOutput, text_inputs
) -> TextConditioningOutput:
    last_hidden_state = outputs.last_hidden_state
    assert last_hidden_state is not None
    attention_mask = text_inputs.attention_mask.to(
        device=last_hidden_state.device, dtype=torch.bool
    )
    seq_lens = [int(x) for x in attention_mask.to(torch.int64).sum(dim=1).tolist()]
    return TextConditioningOutput(last_hidden_state, attention_mask, seq_lens)


@dataclass
class CausalHunyuanVideo15TI2VConfig(PipelineConfig):
    """minWM causal HunyuanVideo 1.5 TI2V (8B, 480p)."""

    task_type: ModelTaskType = ModelTaskType.TI2V
    is_causal: bool = True
    # The shared TI2V branch of InputValidationStage does Wan2.2-style
    # aspect-preserving crop sizing; minWM squash-resizes the image to the
    # requested output size instead (see the pipeline's image prep stage).
    skip_input_image_preprocess: bool = True

    dit_config: DiTConfig = field(default_factory=CausalHunyuanVideo15Config)
    vae_config: VAEConfig = field(default_factory=HunyuanVideo15VAEConfig)

    # FlowMatchDiscreteScheduler shift (SD3 warp).
    flow_shift: float | None = 5.0
    # Clean-context KV-cache refresh runs at timestep 0
    # (minWM ``stabilization_level - 1``).
    context_noise: int = 0

    # Text encoding
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (
            Qwen2_5VLConfig(
                arch_config=QwenImageArchConfig(
                    text_len=QWEN_MAX_LENGTH + PROMPT_CROP_START
                )
            ),
            T5Config(arch_config=T5ArchConfig(text_len=BYT5_MAX_LENGTH)),
        )
    )
    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (qwen_preprocess_text, None)
    )
    postprocess_text_funcs: tuple[Callable[..., TextConditioningOutput], ...] = field(
        default_factory=lambda: (qwen_postprocess_text, byt5_postprocess_text)
    )

    # SigLIP image encoder (729 semantic tokens)
    image_encoder_config: EncoderConfig = field(default_factory=SiglipVisionConfig)
    image_encoder_precision: str = "bf16"

    # Precision
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "bf16")
    )

    def __post_init__(self):
        # First-frame conditioning needs the VAE encoder as well as the decoder.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            keep_resident_components=("dit", "vae"),
        )

    def postprocess_image(self, image_encoder_output: BaseEncoderOutput):
        return image_encoder_output.last_hidden_state

    def preprocess_vae_encode(self, image, vae):
        # ImageVAEEncodingStage pads the condition image with zero pixel
        # frames up to ``num_frames``; minWM encodes only the single image
        # frame and the causal-conv VAE would mix zero frames into the first
        # latent, so drop the padding before encoding.
        return image[:, :, :1]

    def postprocess_image_latent(self, latent_condition, batch):
        # Keep the raw [B, C, 1, h, w] first-frame latent; the denoising stage
        # assembles the channel-concat conditioning tensor itself.
        return latent_condition

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds
