# SPDX-License-Identifier: Apache-2.0
"""Text-only Qwen2.5-VL encoder component.

HunyuanVideo 1.5 ships its text encoder as a bare ``Qwen2_5_VLTextModel``
(no vision tower, no LM head; checkpoint keys are top-level ``embed_tokens.*``
/ ``layers.*`` / ``norm.*``).  This wraps the text backbone from
``qwen2_5vl.py`` in the framework's ``TextEncoder`` contract so the component
loader can resolve the checkpoint's ``architectures: ["Qwen2_5_VLTextModel"]``.
"""

from collections.abc import Iterable

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
)

from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    TextEncoderConfig,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders import qwen2_5vl
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder

_HF_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "attention_dropout",
    "layer_types",
    "sliding_window",
    "use_sliding_window",
    "max_window_layers",
    "bos_token_id",
    "eos_token_id",
    "tie_word_embeddings",
)


class Qwen2_5_VLTextModel(TextEncoder):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__(config)
        arch = config.arch_config
        hf_kwargs = {
            field: getattr(arch, field)
            for field in _HF_CONFIG_FIELDS
            if hasattr(arch, field)
        }
        hf_config = Qwen2_5_VLTextConfig(**hf_kwargs)
        hf_config._attn_implementation = "sdpa"
        self.model = qwen2_5vl.Qwen2_5_VLTextModel(hf_config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=bool(output_hidden_states),
            use_cache=False,
            return_dict=True,
        )
        return BaseEncoderOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attention_mask=attention_mask,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            name = f"model.{name}"
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight.to(param.dtype))
            loaded_params.add(name)
        return loaded_params


EntryClass = [Qwen2_5_VLTextModel]
