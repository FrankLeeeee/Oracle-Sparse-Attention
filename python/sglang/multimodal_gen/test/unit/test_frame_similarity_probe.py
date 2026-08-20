# SPDX-License-Identifier: Apache-2.0

import json
import pathlib

import numpy as np
import torch

from sglang.multimodal_gen.runtime.utils.frame_similarity_probe import (
    FrameSimilarityRecorder,
    frame_pairs,
    pairwise_frame_cosine,
)
from sglang.multimodal_gen.runtime.utils.probe_pass_kind import (
    CACHE_UPDATE_PASS,
    pass_kind_scope,
)

FRAME_SEQLEN = 16
DIM = 8


def test_frame_pairs_are_unordered_and_ordered_stably():
    assert frame_pairs(3) == [(0, 1), (0, 2), (1, 2)]
    assert len(frame_pairs(8)) == 28


def test_pairwise_cosine_matches_a_reference_loop():
    torch.manual_seed(0)
    frames = torch.randn(3, FRAME_SEQLEN, DIM)
    values = pairwise_frame_cosine(frames)
    for index, (i, j) in enumerate(frame_pairs(3)):
        expected = torch.nn.functional.cosine_similarity(
            frames[i].float(), frames[j].float(), dim=-1
        ).mean()
        assert torch.allclose(values[index], expected, atol=1e-6)


def test_pairwise_cosine_is_scale_invariant_per_position():
    """Cosine ignores per-position magnitude, so scaling one frame changes nothing."""
    torch.manual_seed(0)
    frames = torch.randn(2, FRAME_SEQLEN, DIM)
    scaled = frames.clone()
    scaled[1] *= torch.rand(FRAME_SEQLEN, 1) * 10 + 0.1
    assert torch.allclose(
        pairwise_frame_cosine(frames), pairwise_frame_cosine(scaled), atol=1e-5
    )


def test_identical_frames_score_one():
    frame = torch.randn(1, FRAME_SEQLEN, DIM)
    frames = frame.repeat(3, 1, 1)
    assert torch.allclose(pairwise_frame_cosine(frames), torch.ones(3), atol=1e-6)


def _record_video(
    recorder: FrameSimilarityRecorder,
    *,
    chunks: int,
    steps: int,
    layers: int,
    frames_per_block: int = 3,
    groups: int = 1,
) -> None:
    torch.manual_seed(0)
    tokens = groups * frames_per_block * FRAME_SEQLEN
    for chunk in range(chunks):
        for _step in range(steps):
            recorder.begin_forward(
                frame_seqlen=FRAME_SEQLEN,
                num_frames_per_block=frames_per_block,
                query_token_start=chunk * frames_per_block * FRAME_SEQLEN,
            )
            for layer in range(layers + 1):
                recorder.record_layer(
                    layer_index=layer,
                    hidden_states=torch.randn(1, tokens, DIM),
                )
            recorder.end_forward()


def test_flush_writes_a_chunk_step_layer_pair_cube(tmp_path: pathlib.Path):
    recorder = FrameSimilarityRecorder(output_dir=str(tmp_path))
    _record_video(recorder, chunks=4, steps=3, layers=2)
    run_dir = pathlib.Path(recorder.flush(model_tag="Fake", meta={"seed": 42}))

    payload = np.load(run_dir / "frame_similarity.npz")
    # 2 blocks + the final output = 3 layer boundaries; 3 frames = 3 pairs
    assert payload["sim"].shape == (4, 3, 3, 3)
    assert not np.isnan(payload["sim"]).any()
    assert (payload["sim"] <= 1.0 + 1e-5).all()
    assert payload["pairs"].tolist() == [[0, 1], [0, 2], [1, 2]]

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["num_frames_per_block"] == 3
    assert meta["meta"] == {"seed": 42}

    # Flushing is destructive: the next video starts from an empty buffer.
    assert recorder.flush(model_tag="Fake") is None


def test_a_joint_window_is_split_across_its_chunks(tmp_path: pathlib.Path):
    """Rolling Forcing denoises five blocks in one forward, one chunk each."""
    recorder = FrameSimilarityRecorder(output_dir=str(tmp_path))
    _record_video(recorder, chunks=1, steps=1, layers=1, groups=5)
    run_dir = pathlib.Path(recorder.flush(model_tag="Fake"))
    sim = np.load(run_dir / "frame_similarity.npz")["sim"]
    assert sim.shape[0] == 5
    assert not np.isnan(sim).any()


def test_cache_update_pass_is_not_recorded(tmp_path: pathlib.Path):
    recorder = FrameSimilarityRecorder(output_dir=str(tmp_path))
    with pass_kind_scope(CACHE_UPDATE_PASS):
        _record_video(recorder, chunks=2, steps=1, layers=2)
    assert recorder.flush(model_tag="Fake") is None


def test_steps_are_padded_when_chunks_differ(tmp_path: pathlib.Path):
    """Ramp-up chunks take more steps than steady-state ones."""
    recorder = FrameSimilarityRecorder(output_dir=str(tmp_path))
    _record_video(recorder, chunks=1, steps=3, layers=1)
    recorder._chunk_base = 0  # next chunk recorded below
    _record_video(recorder, chunks=1, steps=1, layers=1)
    run_dir = pathlib.Path(recorder.flush(model_tag="Fake"))
    sim = np.load(run_dir / "frame_similarity.npz")["sim"]
    assert sim.shape[1] == 4  # 3 + 1 recorded steps for the same chunk
