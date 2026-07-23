"""LongVie 2 clip-by-clip AR: the pure math the loop stage relies on.

The GPU path is exercised end-to-end by generation runs; these tests pin the
CPU-verifiable invariants — control-pixel slicing, the history-window prefix
that mirrors upstream's chunked VAE encoder, the 36-channel history layout,
and the multi-clip branch of the control encoding stage.
"""

import contextlib
import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    longvie,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longvie import (
    LONGVIE_CONTROL_PIXELS_KEY,
    LONGVIE_TOTAL_NUM_FRAMES_KEY,
    LongVie2ClipLoopStage,
    LongVie2ControlEncodingStage,
)


class _RecordingControlStage:
    """Stands in for LongVie2ControlEncodingStage inside the loop stage."""

    component_name = "vae"
    vae = object()

    def __init__(self):
        self.encoded_shapes = []

    @contextlib.contextmanager
    def use_declared_component(self, *, component_name, module):
        yield module

    def _encode(self, video, *, vae, batch, server_args):
        self.encoded_shapes.append(tuple(video.shape))
        # 4x temporal compression with a leading frame, 16 latent channels
        frames = 1 + (video.shape[2] - 1) // 4
        return torch.arange(
            frames, dtype=torch.float32
        ).view(1, 1, frames, 1, 1).expand(1, 16, frames, 4, 4)


def _make_loop_stage(control_stage=None):
    stub = mock.Mock()
    with mock.patch.object(
        LongVie2ClipLoopStage, "__init__", lambda self, **kwargs: None
    ):
        stage = LongVie2ClipLoopStage()
    stage.control_stage = control_stage if control_stage is not None else stub
    return stage


class TestSliceClip(unittest.TestCase):
    def test_full_slice_is_a_view_of_the_source(self):
        pixels = torch.rand(1, 3, 10, 2, 2)
        clip = LongVie2ClipLoopStage._slice_clip(pixels, 3, 5)
        torch.testing.assert_close(clip, pixels[:, :, 3:8])

    def test_short_tail_is_last_frame_padded(self):
        pixels = torch.rand(1, 3, 10, 2, 2)
        clip = LongVie2ClipLoopStage._slice_clip(pixels, 8, 5)
        self.assertEqual(clip.shape[2], 5)
        torch.testing.assert_close(clip[:, :, :2], pixels[:, :, 8:10])
        for i in range(2, 5):
            torch.testing.assert_close(clip[:, :, i], pixels[:, :, 9])


class TestEncodeHistory(unittest.TestCase):
    def test_only_the_chunkable_prefix_is_encoded(self):
        """Upstream's VAE consumes 1 + 4*((n-1)//4) frames; for n=8 that is 5."""
        control_stage = _RecordingControlStage()
        stage = _make_loop_stage(control_stage)
        history_px = torch.rand(1, 3, 8, 16, 16)

        padded, raw = stage._encode_history(
            history_px, batch=mock.Mock(), server_args=mock.Mock()
        )

        self.assertEqual(control_stage.encoded_shapes, [(1, 3, 5, 16, 16)])
        # 5 pixel frames -> 2 latent frames
        self.assertEqual(raw.shape[2], 2)

    def test_history_latents_are_ones_then_vae(self):
        """Upstream: torch.cat([ones, history_latents], dim=1) — the reverse
        of the control-latent layout."""
        stage = _make_loop_stage(_RecordingControlStage())
        history_px = torch.rand(1, 3, 8, 16, 16)

        padded, raw = stage._encode_history(
            history_px, batch=mock.Mock(), server_args=mock.Mock()
        )

        self.assertEqual(padded.shape[1], 36)
        torch.testing.assert_close(padded[:, :20], torch.ones_like(padded[:, :20]))
        torch.testing.assert_close(padded[:, 20:], raw)


class TestControlStageMultiClip(unittest.TestCase):
    def _forward(self, *, num_frames, clip_num_frames=81, paths=True):
        with mock.patch.object(
            LongVie2ControlEncodingStage, "__init__", lambda self, **kwargs: None
        ):
            stage = LongVie2ControlEncodingStage()

        batch = mock.Mock()
        batch.longvie_dense_video = "dense.mp4" if paths else None
        batch.longvie_sparse_video = "sparse.mp4" if paths else None
        batch.num_frames = num_frames
        batch.height = 16
        batch.width = 16
        batch.extra = {}

        server_args = mock.Mock()
        server_args.pipeline_config.longvie_clip_num_frames = clip_num_frames

        def fake_load(path, *, num_frames, height, width):
            return torch.zeros(1, 3, num_frames, height, width)

        with (
            mock.patch.object(longvie, "load_control_video", side_effect=fake_load),
            mock.patch.object(longvie.pathlib.Path, "is_file", return_value=True),
        ):
            stage.forward(batch, server_args)
        return batch

    def test_multi_clip_clamps_num_frames_and_stashes_pixels(self):
        batch = self._forward(num_frames=161)

        self.assertEqual(batch.num_frames, 81)
        self.assertEqual(batch.extra[LONGVIE_TOTAL_NUM_FRAMES_KEY], 161)
        dense_px, sparse_px = batch.extra[LONGVIE_CONTROL_PIXELS_KEY]
        self.assertEqual(dense_px.shape[2], 161)
        self.assertEqual(sparse_px.shape[2], 161)

    def test_multi_clip_without_controls_raises(self):
        with self.assertRaisesRegex(ValueError, "clip-by-clip"):
            self._forward(num_frames=161, paths=False)


if __name__ == "__main__":
    unittest.main()
