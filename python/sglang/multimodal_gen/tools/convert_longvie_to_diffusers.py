# SPDX-License-Identifier: Apache-2.0
"""Assemble a diffusers-layout LongVie 2 model dir from the upstream release.

LongVie 2 (https://github.com/Vchitect/LongVie) ships two DiffSynth-format
safetensors on top of a stock Wan2.1-I2V-14B-480P:

``dit.safetensors``
    *Only* the self-attention weights (``q``/``k``/``v``/``o`` + ``norm_q``/
    ``norm_k``) of all 40 blocks — 4.2B parameters. Everything else (patch
    embedding, cross-attention, FFN, condition embedders, head) is unchanged
    from the base model, so this is an overlay, not a full checkpoint.

``control.safetensors``
    The dual-control side network: 12 dense + 12 sparse Wan DiT blocks at half
    width (2560) and half heads, plus the zero-initialised fusion linears.

Because the release is an overlay, the base model is *required* to produce
usable weights. Both it and the LongVie release are resolved automatically —
downloaded from the Hub if they are not already local — so the default
invocation needs no paths at all::

    python -m sglang.multimodal_gen.tools.convert_longvie_to_diffusers \\
        --output /data/models/LongVie2-Diffusers

The result is a complete, self-contained diffusers-layout model directory:
the merged transformer (base + finetuned self-attention + control branch)
plus the VAE, text/image encoders, tokenizer and scheduler.
"""

import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.tools.wan_repack import TRANSFORMER_KEYS_RENAME_DICT

logger = init_logger(__name__)

TRANSFORMER_CLASS = "LongVie2Transformer3DModel"
# LongVie 2 ships only a self-attention overlay plus the control branch, so the
# base model is not optional — these are the checkpoints the release is built on
DEFAULT_LONGVIE_REPO = "Vchitect/LongVie2"
DEFAULT_BASE_MODEL = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
LONGVIE_FILES = ("dit.safetensors", "control.safetensors")
# the control branch keys are not part of the Wan naming scheme, so they get
# their own mapping; everything is nested under `control.` in the output so the
# side network is a plain submodule of the transformer
CONTROL_KEYS_RENAME_DICT = {
    "control_blocks_dense": "control.blocks_dense",
    "control_blocks_sparse": "control.blocks_sparse",
    "control_combine_linears": "control.combine_linears",
    "control_initial_combine_linear_dense": "control.in_proj_dense",
    "control_initial_combine_linear_sparse": "control.in_proj_sparse",
    "control_text_linear": "control.text_proj",
    "control_t_mod": "control.time_proj",
}


def rename_wan_key(key: str) -> str:
    """Original-Wan naming -> diffusers naming."""
    for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
        key = key.replace(old, new)
    return key


def rename_control_key(key: str) -> str:
    """Control-branch naming -> `control.*` submodule naming.

    The blocks themselves are ordinary Wan DiT blocks, so their innards go
    through the same Wan rename as the main tower.
    """
    for old, new in CONTROL_KEYS_RENAME_DICT.items():
        if key.startswith(old):
            key = new + key[len(old) :]
            break
    return rename_wan_key(key)


def resolve_model_dir(spec: str, *, allow_patterns: tuple[str, ...] | None = None) -> pathlib.Path:
    """A local directory if it exists, otherwise a Hub repo id to download."""
    path = pathlib.Path(spec).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s from the Hub", spec)
    return pathlib.Path(
        snapshot_download(spec, allow_patterns=list(allow_patterns) if allow_patterns else None)
    )


def load_base_transformer(base_dir: pathlib.Path) -> tuple[dict, dict]:
    """Return the base transformer's (state_dict, config)."""
    transformer_dir = base_dir / "transformer"
    config = json.loads((transformer_dir / "config.json").read_text())
    state_dict: dict[str, torch.Tensor] = {}
    shards = sorted(transformer_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {transformer_dir}")
    for shard in shards:
        state_dict.update(load_file(str(shard)))
    logger.info("Loaded %d base tensors from %d shard(s)", len(state_dict), len(shards))
    return state_dict, config


def overlay_finetuned_attention(
    base_state_dict: dict[str, torch.Tensor],
    dit_path: pathlib.Path,
    *,
    dtype: torch.dtype,
) -> int:
    """Replace the base self-attention weights with LongVie's finetuned ones."""
    overlay = load_file(str(dit_path))
    replaced = 0
    for key, tensor in overlay.items():
        renamed = rename_wan_key(key)
        if renamed not in base_state_dict:
            raise KeyError(
                f"LongVie tensor {key!r} -> {renamed!r} has no counterpart in the "
                "base transformer; the base model is probably not Wan2.1-I2V-14B-480P"
            )
        if base_state_dict[renamed].shape != tensor.shape:
            raise ValueError(
                f"shape mismatch for {renamed}: base "
                f"{tuple(base_state_dict[renamed].shape)} vs LongVie "
                f"{tuple(tensor.shape)}"
            )
        base_state_dict[renamed] = tensor.to(dtype).contiguous()
        replaced += 1
    logger.info("Overlaid %d finetuned self-attention tensors", replaced)
    return replaced


def convert_control_branch(
    control_path: pathlib.Path,
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    control = load_file(str(control_path))
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in control.items():
        renamed = rename_control_key(key)
        if renamed in converted:
            raise ValueError(f"key collision after renaming: {renamed}")
        converted[renamed] = tensor.to(dtype).contiguous()
    logger.info("Converted %d control-branch tensors", len(converted))
    return converted


def build_transformer_config(base_config: dict, control_layers: int) -> dict:
    config = dict(base_config)
    config["_class_name"] = TRANSFORMER_CLASS
    config["control_layers"] = control_layers
    # the side network is half width / half heads, derived here so the model
    # code never has to re-guess it from the checkpoint
    config["control_dim"] = base_config["num_attention_heads"] * base_config[
        "attention_head_dim"
    ] // 2
    config["control_num_heads"] = base_config["num_attention_heads"] // 2
    config["control_ffn_dim"] = base_config["ffn_dim"] // 2
    return config


def copy_base_components(base_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Copy every non-transformer component (VAE, encoders, scheduler, ...)."""
    for child in sorted(base_dir.iterdir()):
        if child.name in {"transformer", "model_index.json"} or child.name.startswith("."):
            continue
        target = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
        logger.info("Copied %s", child.name)


def write_model_index(base_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    index = json.loads((base_dir / "model_index.json").read_text())
    index["_class_name"] = "LongVie2Pipeline"
    index["transformer"] = ["diffusers", TRANSFORMER_CLASS]
    (output_dir / "model_index.json").write_text(json.dumps(index, indent=2, sort_keys=True))


def convert(
    *,
    longvie: str,
    base_model: str,
    output_dir: pathlib.Path,
    dtype: torch.dtype,
    control_layers: int,
) -> None:
    longvie_dir = resolve_model_dir(longvie, allow_patterns=LONGVIE_FILES)
    base_dir = resolve_model_dir(base_model)
    missing = [name for name in LONGVIE_FILES if not (longvie_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{longvie_dir} is missing {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_state_dict, base_config = load_base_transformer(base_dir)
    overlay_finetuned_attention(
        base_state_dict, longvie_dir / "dit.safetensors", dtype=dtype
    )
    base_state_dict.update(
        convert_control_branch(longvie_dir / "control.safetensors", dtype=dtype)
    )

    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    config = build_transformer_config(base_config, control_layers)
    (transformer_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    save_file(
        {k: v.contiguous() for k, v in base_state_dict.items()},
        str(transformer_dir / "diffusion_pytorch_model.safetensors"),
    )
    logger.info("Wrote transformer with %d tensors", len(base_state_dict))

    copy_base_components(base_dir, output_dir)
    write_model_index(base_dir, output_dir)
    verify_complete(output_dir, expected_control_layers=control_layers)
    logger.info("LongVie 2 model directory ready at %s", output_dir)


def verify_complete(output_dir: pathlib.Path, *, expected_control_layers: int) -> None:
    """Fail loudly if the written directory is not a self-contained model.

    The overlay structure makes a half-converted directory easy to produce and
    hard to notice — it loads, and only the outputs are wrong — so the checks
    are on the artifact rather than on the conversion bookkeeping.
    """
    index = json.loads((output_dir / "model_index.json").read_text())
    components = {
        name
        for name, value in index.items()
        if not name.startswith("_") and isinstance(value, (list, tuple))
    }
    missing_dirs = sorted(c for c in components if not (output_dir / c).is_dir())
    if missing_dirs:
        raise FileNotFoundError(
            f"model_index.json declares {sorted(components)} but these have no "
            f"directory: {missing_dirs}"
        )

    weights = load_file(str(output_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    control = {k for k in weights if k.startswith("control.")}
    main = set(weights) - control
    dense_layers = {k.split(".")[2] for k in control if k.startswith("control.blocks_dense.")}
    if len(dense_layers) != expected_control_layers:
        raise ValueError(
            f"expected {expected_control_layers} dense control blocks, found {len(dense_layers)}"
        )
    for required in ("patch_embedding.weight", "proj_out.weight", "scale_shift_table"):
        if required not in main:
            raise KeyError(f"transformer is missing base tensor {required!r}")
    logger.info(
        "Verified: %d transformer tensors (%d base + %d control), components %s",
        len(weights), len(main), len(control), sorted(components),
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longvie", default=DEFAULT_LONGVIE_REPO,
                        help="local dir holding dit/control.safetensors, or a Hub "
                             f"repo id (default: {DEFAULT_LONGVIE_REPO})")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="diffusers-layout base model dir or Hub repo id; "
                             "LongVie 2 only ships a self-attention overlay, so the "
                             f"base is required (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--control-layers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    convert(
        longvie=args.longvie,
        base_model=args.base_model,
        output_dir=args.output,
        dtype=dtype,
        control_layers=args.control_layers,
    )


if __name__ == "__main__":
    main()
