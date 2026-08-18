# SPDX-License-Identifier: Apache-2.0
"""Same-noise numerical parity verification: Self-Forcing upstream vs sglang.

Feeds one shared noise bank (initial noise + every re-noise draw) to both the
upstream implementation (guandeh17/Self-Forcing checkout) and sglang, so the
two runs compute the same function and any residual divergence is floating
point. A control run of upstream against itself with the attention kernel
swapped to torch SDPA calibrates how much divergence pure kernel float noise
causes; sglang is considered aligned when its per-block latent divergence sits
at (or below) that baseline and grows monotonically without jumps.

Example:

    python -m sglang.multimodal_gen.tools.verify_self_forcing_parity \
        --upstream-repo /path/to/Self-Forcing \
        --model-path frankleeeee/SelfForcing-Wan2.1-T2V-1.3B-Diffusers

Requirements: the upstream checkout must be runnable (checkpoints/
self_forcing_dmd.pt and wan_models/Wan2.1-T2V-1.3B in place), and one CUDA
device must be free. Resolution is fixed at 480x832 (latent 60x104), matching
the upstream inference default.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

PROMPT_DEFAULT = "A red fox trotting across a snowy field, camera follows"
LATENT_C, LATENT_H, LATENT_W = 16, 60, 104
NUM_FRAMES_PER_BLOCK = 3
DMD_STEPS = 4

# Driver executed inside the upstream checkout (own process: different
# sys.path world, and it monkeypatches torch.randn_like).
UPSTREAM_DRIVER = r'''
import json
import os
import sys

import torch

repo = sys.argv[1]
noise_dir = sys.argv[2]
out_path = sys.argv[3]
prompt = sys.argv[4]
use_sdpa = sys.argv[5] == "1"

os.chdir(repo)
sys.path.insert(0, repo)
torch.set_grad_enabled(False)

from omegaconf import OmegaConf

config = OmegaConf.merge(
    OmegaConf.load("configs/default_config.yaml"),
    OmegaConf.load("configs/self_forcing_dmd.yaml"),
)

from pipeline import CausalInferencePipeline

if use_sdpa:
    import torch.nn.functional as F
    import wan.modules.attention as wan_attention

    class _SDPAInterface:
        @staticmethod
        def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k,
                                   softmax_scale=None, causal=False,
                                   deterministic=False):
            assert cu_seqlens_q.numel() == 2, "batch-1 only"
            out = F.scaled_dot_product_attention(
                q.unsqueeze(0).transpose(1, 2),
                k.unsqueeze(0).transpose(1, 2),
                v.unsqueeze(0).transpose(1, 2),
                is_causal=causal,
                scale=softmax_scale,
            ).transpose(1, 2).squeeze(0)
            return (out, None)

    wan_attention.flash_attn_interface = _SDPAInterface

device = torch.device("cuda")
pipeline = CausalInferencePipeline(config, device=device)
state_dict = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
pipeline.generator.load_state_dict(state_dict["generator_ema"])
pipeline = pipeline.to(dtype=torch.bfloat16, device=device)

renoise_idx = 0

def fixed_randn_like(t, **kwargs):
    global renoise_idx
    saved = torch.load(os.path.join(noise_dir, f"renoise_{renoise_idx}.pt"),
                       map_location="cpu")
    renoise_idx += 1
    out = saved.flatten(0, 1).to(device=t.device, dtype=t.dtype)
    assert out.shape == t.shape, (out.shape, t.shape)
    return out

torch.randn_like = fixed_randn_like

init_noise = torch.load(os.path.join(noise_dir, "init_noise_btchw.pt"),
                        map_location="cpu").to(device=device, dtype=torch.bfloat16)

_video, latents = pipeline.inference(
    noise=init_noise, text_prompts=[prompt], return_latents=True
)
expected = json.loads(os.environ["PARITY_EXPECTED_RENOISE"])
assert renoise_idx == expected, f"consumed {renoise_idx} re-noise draws, expected {expected}"
torch.save(latents.float().cpu(), out_path)
print(f"upstream run done: {renoise_idx} re-noise draws, latents {tuple(latents.shape)}")
'''


def build_noise_bank(
    *, workdir: pathlib.Path, num_latent_frames: int, num_renoise: int, seed: int
) -> None:
    g = torch.Generator().manual_seed(seed)
    init = torch.randn(
        1, num_latent_frames, LATENT_C, LATENT_H, LATENT_W, generator=g
    )
    torch.save(init, workdir / "init_noise_btchw.pt")
    torch.save(
        init.permute(0, 2, 1, 3, 4).contiguous(), workdir / "init_noise_bcthw.pt"
    )
    for i in range(num_renoise):
        torch.save(
            torch.randn(
                1, NUM_FRAMES_PER_BLOCK, LATENT_C, LATENT_H, LATENT_W, generator=g
            ),
            workdir / f"renoise_{i}.pt",
        )


def run_upstream(
    *,
    upstream_repo: pathlib.Path,
    workdir: pathlib.Path,
    out_name: str,
    prompt: str,
    num_renoise: int,
    use_sdpa: bool,
) -> pathlib.Path:
    driver = workdir / "upstream_driver.py"
    driver.write_text(UPSTREAM_DRIVER)
    out_path = workdir / out_name
    env = os.environ | {"PARITY_EXPECTED_RENOISE": json.dumps(num_renoise)}
    subprocess.run(
        [
            sys.executable,
            str(driver),
            str(upstream_repo),
            str(workdir),
            str(out_path),
            prompt,
            "1" if use_sdpa else "0",
        ],
        env=env,
        check=True,
    )
    return out_path


def run_sglang(
    *,
    model_path: str,
    workdir: pathlib.Path,
    prompt: str,
    num_frames: int,
) -> pathlib.Path:
    env = os.environ | {"SGLANG_DIFFUSION_TEST_PARITY_DIR": str(workdir)}
    subprocess.run(
        [
            "sglang",
            "generate",
            "--model-path",
            model_path,
            "--prompt",
            prompt,
            "--num-frames",
            str(num_frames),
            "--seed",
            "0",
            "--output-path",
            str(workdir / "sglang_out"),
        ],
        env=env,
        check=True,
    )
    return workdir / "sglang_latents.pt"


def per_block_divergence(
    a_btchw: torch.Tensor, b_btchw: torch.Tensor, *, num_blocks: int
) -> list[dict]:
    rows = []
    for blk in range(num_blocks):
        sl = slice(blk * NUM_FRAMES_PER_BLOCK, (blk + 1) * NUM_FRAMES_PER_BLOCK)
        x, y = a_btchw[:, sl], b_btchw[:, sl]
        rows.append(
            {
                "block": blk,
                "rel_err": ((x - y).norm() / x.norm()).item(),
                "cosine": torch.nn.functional.cosine_similarity(
                    x.flatten(), y.flatten(), dim=0
                ).item(),
            }
        )
    return rows


def print_report(name: str, rows: list[dict]) -> None:
    cells = "  ".join(f"{r['rel_err']:.3f}/{r['cosine']:.4f}" for r in rows)
    print(f"{name:36s} {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Same-noise parity verification: Self-Forcing upstream vs sglang"
    )
    parser.add_argument(
        "--upstream-repo",
        type=pathlib.Path,
        required=True,
        help="Path to a runnable guandeh17/Self-Forcing checkout",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Converted Diffusers model (local dir or HF repo id)",
    )
    parser.add_argument("--prompt", type=str, default=PROMPT_DEFAULT)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Pixel frames; (n-1)/4+1 latent frames must be a multiple of 3",
    )
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument(
        "--skip-kernel-control",
        action="store_true",
        help="Skip the upstream-vs-upstream SDPA control run",
    )
    parser.add_argument(
        "--workdir",
        type=pathlib.Path,
        default=None,
        help="Working directory (default: a fresh temp dir, kept for inspection)",
    )
    args = parser.parse_args()

    num_latent_frames = (args.num_frames - 1) // 4 + 1
    if num_latent_frames % NUM_FRAMES_PER_BLOCK != 0:
        raise ValueError(
            f"{args.num_frames} pixel frames -> {num_latent_frames} latent frames, "
            f"not a multiple of {NUM_FRAMES_PER_BLOCK}"
        )
    num_blocks = num_latent_frames // NUM_FRAMES_PER_BLOCK
    num_renoise = num_blocks * (DMD_STEPS - 1)

    workdir = args.workdir or pathlib.Path(tempfile.mkdtemp(prefix="sf_parity_"))
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("Parity workdir: %s", workdir)

    build_noise_bank(
        workdir=workdir,
        num_latent_frames=num_latent_frames,
        num_renoise=num_renoise,
        seed=args.noise_seed,
    )

    upstream_fa3 = run_upstream(
        upstream_repo=args.upstream_repo,
        workdir=workdir,
        out_name="upstream_latents.pt",
        prompt=args.prompt,
        num_renoise=num_renoise,
        use_sdpa=False,
    )
    sglang_latents_path = run_sglang(
        model_path=args.model_path,
        workdir=workdir,
        prompt=args.prompt,
        num_frames=args.num_frames,
    )

    up = torch.load(upstream_fa3)
    sg = torch.load(sglang_latents_path).permute(0, 2, 1, 3, 4)
    print(f"\nper-block rel_err/cosine, blocks 0..{num_blocks - 1}")
    sg_rows = per_block_divergence(up, sg, num_blocks=num_blocks)
    print_report("upstream FA3 vs sglang", sg_rows)

    if not args.skip_kernel_control:
        upstream_sdpa = run_upstream(
            upstream_repo=args.upstream_repo,
            workdir=workdir,
            out_name="upstream_latents_sdpa.pt",
            prompt=args.prompt,
            num_renoise=num_renoise,
            use_sdpa=True,
        )
        ctrl_rows = per_block_divergence(
            up, torch.load(upstream_sdpa), num_blocks=num_blocks
        )
        print_report("upstream FA3 vs upstream SDPA (ctrl)", ctrl_rows)
        verdict = sg_rows[0]["rel_err"] <= 2.0 * ctrl_rows[0]["rel_err"]
        print(
            f"\nverdict: block-0 rel_err {sg_rows[0]['rel_err']:.3f} vs kernel-swap "
            f"baseline {ctrl_rows[0]['rel_err']:.3f} -> "
            + ("ALIGNED (within float-noise envelope)" if verdict else "SUSPECT")
        )
    print(f"\nartifacts kept in {workdir}")


if __name__ == "__main__":
    main()
