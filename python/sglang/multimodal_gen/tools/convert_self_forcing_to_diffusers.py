# SPDX-License-Identifier: Apache-2.0
"""Convert the Self-Forcing training checkpoint to Diffusers layout.

Self-Forcing (guandeh17/Self-Forcing, https://huggingface.co/gdhe17/Self-Forcing)
releases a DiT-only ``.pt`` training state (keys ``generator`` /
``generator_ema``, original Wan parameter naming) on top of Wan2.1-T2V-1.3B.
This entry point assembles the same self-contained Diffusers-style directory
as convert_forcing_to_diffusers.py (whose machinery it reuses), with the
geometry of the released checkpoint baked in:

The released self_forcing_dmd.pt pairs with the upstream default config
(``local_attn_size=-1``): full-context attention with no rolling window —
upstream's fixed 32760-token cache is simply the whole 21-latent-frame video.
``sliding_window_num_frames=None`` makes our runtime size the KV cache to the
whole video and never evict. Upstream's local21/720p configs
(``local_attn_size=21``) belong to different checkpoints.

Example:

    python -m sglang.multimodal_gen.tools.convert_self_forcing_to_diffusers \
        --checkpoint /tmp/self-forcing-upstream/checkpoints/self_forcing_dmd.pt \
        --output-path /tmp/SelfForcing-Wan2.1-T2V-1.3B-Diffusers
"""

import argparse
import pathlib

import torch

from sglang.multimodal_gen.tools.convert_forcing_to_diffusers import convert

SELF_FORCING_PRESET: dict = {
    "pipeline_class": "CausalForcingPipeline",
    "transformer_class": "CausalWanTransformer3DModel",
    "arch_overrides": {
        "num_frames_per_block": 3,
        "sliding_window_num_frames": None,
        "sink_size": 0,
    },
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Self-Forcing checkpoint to Diffusers layout"
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to the upstream self_forcing_dmd.pt training checkpoint",
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
        preset=SELF_FORCING_PRESET,
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        output_path=args.output_path,
        prefer_ema=not args.no_ema,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
    )


if __name__ == "__main__":
    main()
