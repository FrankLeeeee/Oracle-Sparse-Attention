# SPDX-License-Identifier: Apache-2.0
"""Frame-to-frame pattern similarity from the raw Q/K dumps of run.py.

    python similarity.py [--run p1] [--device cuda]

For a captured attention call with Sq query tokens and Sk key tokens, each
latent frame being T tokens, the map splits into (Sq/T) x (Sk/T) frame-to-frame
pairs. For query frame i and key frame j the pair map is

    A_ij = softmax(Q_i @ K_j^T / sqrt(d))   # [T, T], softmax per query row
                                            # over that key frame only

i.e. the *shape* of frame i's attention into frame j, independent of how much
total mass frame j receives. The reported score is the cosine similarity of
the flattened A_i0 and A_ij — column j says how similar frame i's pattern into
key frame j is to its pattern into key frame 0 (so column 0 is 1 by
construction). High values across j are the empirical premise behind OSA's
replicate design: one frame-to-frame pattern per head, reused for every
history frame.

Output: results/investigation/qk_map_similarity/similarity/<run>/sim_L{l}_h{h}_c{c}_s{s}.json
"""

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import results_dir  # noqa: E402

from plot_maps import load_qk  # noqa: E402
from run import CHUNK_IDS, SPEC_SETS, STEP_IDS  # noqa: E402

ROOT = results_dir("qk_map_similarity")


@torch.no_grad()
def pair_softmax(
    query_frame: torch.Tensor, key_frame: torch.Tensor
) -> torch.Tensor:
    """``softmax(Q_i K_j^T / sqrt(d))`` over this key frame only -> [T, T]."""
    scale = query_frame.shape[-1] ** -0.5
    return torch.softmax((query_frame @ key_frame.T) * scale, dim=-1)


@torch.no_grad()
def self_references(
    query: torch.Tensor, key: torch.Tensor, *, frame_seqlen: int
) -> list[torch.Tensor]:
    """Normalized flattened ``A_{i, self}`` per query frame i.

    Query frame i of a chunk is the (num_key_frames - 3 + i)-th key frame —
    the chunk's own frames are the newest 3 of the visible cache — so this is
    the map of each frame attending to itself.
    """
    key_frames = key.shape[0] // frame_seqlen
    references = []
    for i in range(query.shape[0] // frame_seqlen):
        q_frame = query[i * frame_seqlen : (i + 1) * frame_seqlen]
        j_self = key_frames - 3 + i
        k_frame = key[j_self * frame_seqlen : (j_self + 1) * frame_seqlen]
        flat = pair_softmax(q_frame, k_frame).flatten()
        references.append(flat / flat.norm())
    return references


@torch.no_grad()
def cosine_table(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    frame_seqlen: int,
    device: str,
    references: list[torch.Tensor] | None = None,
) -> list[list[float]]:
    """``cos(reference_i, A_ij)`` for every query frame i and key frame j.

    ``references`` defaults to this call's own self maps (``A_{i, -3+i}``);
    the temporal-consistency analysis passes another chunk's self maps in.
    """
    query, key = query.to(device), key.to(device)
    if references is None:
        references = self_references(query, key, frame_seqlen=frame_seqlen)
    key_frames = key.shape[0] // frame_seqlen
    table = []
    for i in range(query.shape[0] // frame_seqlen):
        q_frame = query[i * frame_seqlen : (i + 1) * frame_seqlen]
        reference = references[i].to(device)
        row = []
        for j in range(key_frames):
            k_frame = key[j * frame_seqlen : (j + 1) * frame_seqlen]
            pair = pair_softmax(q_frame, k_frame).flatten()
            row.append(float(reference @ (pair / pair.norm())))
        table.append(row)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="p1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--spec", default="main", choices=sorted(SPEC_SETS))
    parser.add_argument(
        "--ref-chunk",
        type=int,
        default=0,
        help="reference chunk C of the temporal-consistency pass",
    )
    parser.add_argument("--temporal-step", type=int, default=3)
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run
    out_dir = ROOT / "similarity" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in SPEC_SETS[args.spec]:
        for chunk in CHUNK_IDS:
            for step in STEP_IDS:
                query, key, frame_seqlen = load_qk(
                    run_dir, spec["layer"], spec["head"], chunk, step
                )
                table = cosine_table(
                    query, key, frame_seqlen=frame_seqlen, device=args.device
                )
                record = {
                    "run": args.run,
                    "task": spec["task"],
                    "layer": spec["layer"],
                    "head": spec["head"],
                    "chunk": chunk,
                    "step": step,
                    "frame_seqlen": frame_seqlen,
                    "query_frames": len(table),
                    "key_frames": len(table[0]),
                    "reference": "self",
                    "self_columns": [
                        len(table[0]) - 3 + i for i in range(len(table))
                    ],
                    "cosine": [[round(v, 4) for v in row] for row in table],
                }
                name = f"sim_L{spec['layer']:02d}_h{spec['head']}_c{chunk}_s{step}.json"
                (out_dir / name).write_text(json.dumps(record, indent=2))
            print(f"[sim] L{spec['layer']} h{spec['head']} c{chunk}: done", flush=True)

        # Temporal consistency: reference the self maps A_{C,i,-3+i} of chunk
        # C, compare against every captured chunk's maps A_{c,i,j}.
        query, key, frame_seqlen = load_qk(
            run_dir, spec["layer"], spec["head"], args.ref_chunk, args.temporal_step
        )
        refs = self_references(
            query.to(args.device), key.to(args.device), frame_seqlen=frame_seqlen
        )
        chunks = {}
        for chunk in CHUNK_IDS:
            query, key, frame_seqlen = load_qk(
                run_dir, spec["layer"], spec["head"], chunk, args.temporal_step
            )
            table = cosine_table(
                query,
                key,
                frame_seqlen=frame_seqlen,
                device=args.device,
                references=refs,
            )
            chunks[chunk] = [[round(v, 4) for v in row] for row in table]
        record = {
            "run": args.run,
            "layer": spec["layer"],
            "head": spec["head"],
            "ref_chunk": args.ref_chunk,
            "step": args.temporal_step,
            "chunks": chunks,
        }
        name = (
            f"temporal_L{spec['layer']:02d}_h{spec['head']}"
            f"_ref{args.ref_chunk}_s{args.temporal_step}.json"
        )
        (out_dir / name).write_text(json.dumps(record, indent=2))
        print(f"[sim] L{spec['layer']} h{spec['head']} temporal: done", flush=True)
    print(f"[sim] wrote {len(list(out_dir.glob('*.json')))} tables -> {out_dir}")


if __name__ == "__main__":
    main()
