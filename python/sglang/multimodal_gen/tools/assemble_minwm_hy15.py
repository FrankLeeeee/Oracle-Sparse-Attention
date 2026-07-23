# SPDX-License-Identifier: Apache-2.0
"""Assemble a minWM causal HunyuanVideo 1.5 model dir in Diffusers layout.

MIN-Lab/minWM publishes the HY15 TI2V transformers as DiT-only diffusers
folders (``HY15/TI2V/{bidirectional,ar_diffusion_tf,causal_ode,causal_cd,dmd}``
— fp32 safetensors with original tencent parameter naming), while the VAE and
encoder stack come from the stock HunyuanVideo-1.5 checkpoint
(``hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v``).

This tool combines the two into a self-contained model directory that the
SGLang diffusion runtime loads directly:

    <output>/
        model_index.json     (_class_name = CausalHunyuanVideo15Pipeline)
        transformer/         (minWM weights, cast to bf16 by default)
        vae/ text_encoder/ tokenizer/ text_encoder_2/ tokenizer_2/
        image_encoder/ feature_extractor/ scheduler/     (copied from base)

Example:

    python -m sglang.multimodal_gen.tools.assemble_minwm_hy15 \
        --minwm-transformer /data/models/_downloads/minWM/HY15/TI2V/dmd \
        --hy15-base /data/models/_downloads/HY15-base \
        --output-path /data/models/minWM-HY15-TI2V-dmd-Diffusers
"""

import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_BASE_COMPONENTS = (
    "vae",
    "text_encoder",
    "tokenizer",
    "text_encoder_2",
    "tokenizer_2",
    "image_encoder",
    "feature_extractor",
    "scheduler",
)

_MODEL_INDEX = {
    "_class_name": "CausalHunyuanVideo15Pipeline",
    "_diffusers_version": "0.36.0.dev0",
    "transformer": ["diffusers", "CausalHunyuanVideo15Transformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLHunyuanVideo15"],
    "text_encoder": ["transformers", "Qwen2_5_VLTextModel"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "tokenizer_2": ["transformers", "ByT5Tokenizer"],
    "image_encoder": ["transformers", "SiglipVisionModel"],
    "feature_extractor": ["transformers", "SiglipImageProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
}

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _copy_base_components(base: pathlib.Path, output: pathlib.Path) -> None:
    for component in _BASE_COMPONENTS:
        src = base / component
        if not src.is_dir():
            raise FileNotFoundError(f"missing base component: {src}")
        dst = output / component
        if dst.exists():
            logger.info("Skipping %s (already present)", component)
            continue
        logger.info("Copying %s ...", component)
        shutil.copytree(src, dst)


def _write_transformer(
    minwm_transformer: pathlib.Path,
    output: pathlib.Path,
    dtype: torch.dtype,
) -> None:
    dst = output / "transformer"
    dst.mkdir(parents=True, exist_ok=True)

    config = json.loads((minwm_transformer / "config.json").read_text())
    # The runtime resolves the transformer class from ``_class_name``; minWM's
    # original class name is kept as provenance.
    config["_original_class_name"] = config.get("_class_name")
    config["_class_name"] = "CausalHunyuanVideo15Transformer3DModel"
    (dst / "config.json").write_text(json.dumps(config, indent=4) + "\n")

    weights_file = minwm_transformer / "diffusion_pytorch_model.safetensors"
    logger.info("Loading %s ...", weights_file)
    state_dict = load_file(str(weights_file))
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    out_file = dst / "diffusion_pytorch_model.safetensors"
    logger.info("Saving %d tensors to %s ...", len(state_dict), out_file)
    save_file(state_dict, str(out_file))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minwm-transformer",
        type=pathlib.Path,
        required=True,
        help="minWM HY15 TI2V transformer dir "
        "(e.g. <minWM>/HY15/TI2V/dmd with config.json + safetensors)",
    )
    parser.add_argument(
        "--hy15-base",
        type=pathlib.Path,
        required=True,
        help="Diffusers-layout HunyuanVideo-1.5 checkpoint providing the "
        "VAE/encoder stack (hunyuanvideo-community/"
        "HunyuanVideo-1.5-Diffusers-480p_i2v)",
    )
    parser.add_argument("--output-path", type=pathlib.Path, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=sorted(_DTYPES))
    args = parser.parse_args()

    output: pathlib.Path = args.output_path
    output.mkdir(parents=True, exist_ok=True)

    (output / "model_index.json").write_text(
        json.dumps(_MODEL_INDEX, indent=2) + "\n"
    )
    _copy_base_components(args.hy15_base, output)
    _write_transformer(args.minwm_transformer, output, _DTYPES[args.dtype])
    logger.info("Assembled model dir at %s", output)


if __name__ == "__main__":
    main()
