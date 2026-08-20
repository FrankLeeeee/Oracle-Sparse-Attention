# SPDX-License-Identifier: Apache-2.0

import json
import pathlib

import pytest
import torch

from sglang.multimodal_gen.runtime.utils.chunk_timing_probe import (
    CROSS_ATTENTION,
    SELF_ATTENTION,
    ChunkTimingRecorder,
)
from sglang.multimodal_gen.runtime.utils.probe_pass_kind import (
    CACHE_UPDATE_PASS,
    DENOISE_PASS,
    current_pass_kind,
    pass_kind_scope,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the probe times CUDA events"
)


def _busy(device: torch.device, size: int = 512) -> None:
    left = torch.randn(size, size, device=device)
    right = torch.randn(size, size, device=device)
    left @ right


def test_pass_kind_scope_nests_and_restores():
    assert current_pass_kind() == DENOISE_PASS
    with pass_kind_scope(CACHE_UPDATE_PASS):
        assert current_pass_kind() == CACHE_UPDATE_PASS
    assert current_pass_kind() == DENOISE_PASS


def test_recording_scope_only_narrows(tmp_path: pathlib.Path):
    recorder = ChunkTimingRecorder(output_dir=str(tmp_path))
    with recorder.recording_scope(False):
        with recorder.recording_scope(True):
            assert recorder.enabled is False
    assert recorder.enabled is True


def test_flush_without_records_writes_nothing(tmp_path: pathlib.Path):
    recorder = ChunkTimingRecorder(output_dir=str(tmp_path))
    assert recorder.flush(model_tag="Fake") is None
    assert not list(tmp_path.iterdir())


@requires_cuda
def test_flush_groups_regions_by_chunk_and_pass(tmp_path: pathlib.Path):
    device = torch.device("cuda")
    recorder = ChunkTimingRecorder(output_dir=str(tmp_path))
    recorder.note_layer_count(2)
    for chunk in (0, 1):
        for _step in range(3):
            recorder.begin_forward(chunk_index=chunk)
            for _layer in range(2):
                with recorder.region(SELF_ATTENTION):
                    _busy(device)
                with recorder.region(CROSS_ATTENTION):
                    _busy(device, size=128)
            recorder.end_forward()
        with pass_kind_scope(CACHE_UPDATE_PASS):
            recorder.begin_forward(chunk_index=chunk)
            with recorder.region(SELF_ATTENTION):
                _busy(device)
            recorder.end_forward()

    run_dir = recorder.flush(model_tag="Fake", meta={"prompt": "x"})
    payload = json.loads((pathlib.Path(run_dir) / "chunk_timing.json").read_text())

    assert payload["num_layers"] == 2
    assert payload["meta"] == {"prompt": "x"}
    assert [entry["chunk"] for entry in payload["chunks"]] == [0, 1]
    for entry in payload["chunks"]:
        denoise = entry[DENOISE_PASS]
        assert denoise["steps"] == 3
        assert len(denoise["forward_ms_per_step"]) == 3
        # Attention is timed inside the forward it belongs to.
        attention = denoise["self_attn_ms"] + denoise["cross_attn_ms"]
        assert 0 < attention <= denoise["forward_ms"]
        assert entry[CACHE_UPDATE_PASS]["steps"] == 1

    # Flushing is destructive: the next video starts from an empty buffer.
    assert recorder.flush(model_tag="Fake") is None


@requires_cuda
def test_regions_outside_a_forward_are_dropped(tmp_path: pathlib.Path):
    recorder = ChunkTimingRecorder(output_dir=str(tmp_path))
    with recorder.region(SELF_ATTENTION):
        _busy(torch.device("cuda"))
    assert recorder.flush(model_tag="Fake") is None


@requires_cuda
def test_disabled_scope_skips_recording(tmp_path: pathlib.Path):
    recorder = ChunkTimingRecorder(output_dir=str(tmp_path))
    with recorder.recording_scope(False):
        recorder.begin_forward(chunk_index=0)
        with recorder.region(SELF_ATTENTION):
            _busy(torch.device("cuda"))
        recorder.end_forward()
    assert recorder.flush(model_tag="Fake") is None
