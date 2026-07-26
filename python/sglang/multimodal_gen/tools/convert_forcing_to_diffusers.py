# SPDX-License-Identifier: Apache-2.0
"""Convert Causal-Forcing / Rolling Forcing training checkpoints to Diffusers layout.

Both projects release DiT-only ``.pt`` training states (keys ``generator`` /
``generator_ema`` with ``model.[_fsdp_wrapped_module.]`` prefixes, original Wan
parameter naming) on top of Wan2.1-T2V-1.3B:

- Causal-Forcing (thu-ml): https://huggingface.co/zhuhz22/Causal-Forcing
- Rolling Forcing (TencentARC): https://huggingface.co/TencentARC/RollingForcing

This tool assembles a self-contained Diffusers-style model directory that the
SGLang diffusion runtime can load directly:

    <output>/
        model_index.json            (_class_name = pipeline preset)
        transformer/                (converted weights + causal config)
        scheduler/ text_encoder/ tokenizer/ vae/   (copied from the Wan base)

Example:

    python -m sglang.multimodal_gen.tools.convert_forcing_to_diffusers \
        --preset causal-forcing-chunkwise \
        --checkpoint ~/.cache/huggingface/hub/models--zhuhz22--Causal-Forcing/snapshots/<hash>/chunkwise/causal_forcing.pt \
        --output-path /data/models/CausalForcing-Wan2.1-T2V-1.3B-chunkwise-Diffusers
"""

import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.tools.wan_repack import TRANSFORMER_KEYS_RENAME_DICT

logger = init_logger(__name__)

_STATE_DICT_PREFIXES = ("model._fsdp_wrapped_module.", "model.")

# Per-preset pipeline/transformer classes and causal attention geometry
# (latent-frame units). These mirror the upstream inference configs.
PRESETS: dict[str, dict] = {
    "causal-forcing-chunkwise": {
        "pipeline_class": "CausalForcingPipeline",
        "transformer_class": "CausalWanTransformer3DModel",
        "arch_overrides": {
            "num_frames_per_block": 3,
            "sliding_window_num_frames": 21,
            "sink_size": 0,
        },
    },
    "causal-forcing-framewise": {
        "pipeline_class": "CausalForcingPipeline",
        "transformer_class": "CausalWanTransformer3DModel",
        "arch_overrides": {
            "num_frames_per_block": 1,
            "sliding_window_num_frames": 21,
            "sink_size": 0,
        },
    },
    "self-forcing": {
        # Self-Forcing (guandeh17/Self-Forcing) — the ancestor of Causal
        # Forcing; same chunk-wise DMD inference geometry (3-frame blocks,
        # 21-latent-frame context, no pinned sink), different weights.
        "pipeline_class": "CausalForcingPipeline",
        "transformer_class": "CausalWanTransformer3DModel",
        "arch_overrides": {
            "num_frames_per_block": 3,
            "sliding_window_num_frames": 21,
            "sink_size": 0,
        },
    },
    "rolling-forcing": {
        "pipeline_class": "WanRollingForcingPipeline",
        "transformer_class": "RollingForcingWanTransformer3DModel",
        "arch_overrides": {
            "num_frames_per_block": 3,
            # Upstream: KV cache buffer of 24 latent frames, attention context
            # capped at 21 frames, first 3-frame block pinned as attention sink.
            "sliding_window_num_frames": 24,
            "max_attention_num_frames": 21,
            "sink_size": 3,
        },
    },
}


def _strip_prefix(key: str) -> str:
    for prefix in _STATE_DICT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _rename_to_diffusers(key: str) -> str:
    for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
        key = key.replace(old, new)
    return key


def load_generator_state_dict(
    checkpoint_path: pathlib.Path,
    *,
    prefer_ema: bool,
) -> dict[str, torch.Tensor]:
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    candidates = ["generator_ema", "generator"] if prefer_ema else ["generator"]
    for candidate in candidates:
        if candidate in state:
            logger.info("Using '%s' weights from %s", candidate, checkpoint_path)
            return state[candidate]
    raise KeyError(
        f"None of {candidates} found in checkpoint; got top-level keys {list(state)}"
    )


def convert_transformer_weights(
    raw_state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in raw_state_dict.items():
        new_key = _rename_to_diffusers(_strip_prefix(key))
        if new_key in converted:
            raise ValueError(f"Key collision after renaming: {new_key}")
        converted[new_key] = tensor.to(dtype).contiguous()
    return converted


def write_transformer(
    converted_state_dict: dict[str, torch.Tensor],
    *,
    base_transformer_config: dict,
    preset: dict,
    output_path: pathlib.Path,
) -> None:
    transformer_dir = output_path / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    config = dict(base_transformer_config)
    config["_class_name"] = preset["transformer_class"]
    config.update(preset["arch_overrides"])
    with open(transformer_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    save_file(
        converted_state_dict,
        str(transformer_dir / "diffusion_pytorch_model.safetensors"),
    )


def copy_base_components(
    base_model_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    for component in ("scheduler", "text_encoder", "tokenizer", "vae"):
        src = base_model_path / component
        dst = output_path / component
        if dst.exists():
            logger.info("Skipping existing component %s", dst)
            continue
        # symlinks=False resolves the HF-cache blob symlinks into real files so
        # the output directory is self-contained.
        shutil.copytree(src, dst, symlinks=False)


def write_model_index(
    base_model_path: pathlib.Path,
    preset: dict,
    output_path: pathlib.Path,
) -> None:
    with open(base_model_path / "model_index.json") as f:
        model_index = json.load(f)
    model_index["_class_name"] = preset["pipeline_class"]
    model_index["transformer"] = ["diffusers", preset["transformer_class"]]
    with open(output_path / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)


def resolve_base_model_path(base_model: str) -> pathlib.Path:
    path = pathlib.Path(base_model).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(base_model))


def convert(
    *,
    preset_name: str,
    checkpoint: pathlib.Path,
    base_model: str,
    output_path: pathlib.Path,
    prefer_ema: bool,
    dtype: torch.dtype,
) -> None:
    preset = PRESETS[preset_name]
    base_model_path = resolve_base_model_path(base_model)

    raw_state_dict = load_generator_state_dict(checkpoint, prefer_ema=prefer_ema)
    converted = convert_transformer_weights(raw_state_dict, dtype=dtype)
    logger.info("Converted %d transformer tensors", len(converted))

    with open(base_model_path / "transformer" / "config.json") as f:
        base_transformer_config = json.load(f)

    output_path.mkdir(parents=True, exist_ok=True)
    write_transformer(
        converted,
        base_transformer_config=base_transformer_config,
        preset=preset,
        output_path=output_path,
    )
    copy_base_components(base_model_path, output_path)
    write_model_index(base_model_path, preset, output_path)
    logger.info("Done. Converted model written to %s", output_path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Causal-Forcing / Rolling Forcing checkpoints to Diffusers layout"
    )
    parser.add_argument(
        "--preset", type=str, required=True, choices=sorted(PRESETS.keys())
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to the upstream .pt training checkpoint",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Wan base model (local Diffusers directory or HF repo id)",
    )
    parser.add_argument("--output-path", type=pathlib.Path, required=True)
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use raw 'generator' weights instead of 'generator_ema'",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    return parser.parse_args()


def main() -> None:
    args = get_args()
    convert(
        preset_name=args.preset,
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        output_path=args.output_path,
        prefer_ema=not args.no_ema,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
    )


if __name__ == "__main__":
    main()
