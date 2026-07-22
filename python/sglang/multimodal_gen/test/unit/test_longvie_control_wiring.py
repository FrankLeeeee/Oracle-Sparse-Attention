"""LongVie 2 control-path wiring: CLI -> SamplingParams -> Req.

``Req`` declares ``longvie_dense_video`` / ``longvie_sparse_video`` as its own
fields, so they shadow the sampling params rather than being delegated to them.
That makes ``apply_request_extra`` load-bearing: without it the control paths
silently stay ``None`` and the pipeline degrades to plain Wan I2V.
"""

import argparse
import unittest

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.wan import (
    LongVie2SamplingParams,
    WanI2V_14B_480P_SamplingParam,
)
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class TestLongVie2ControlWiring(unittest.TestCase):
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        return parser.parse_args(argv)

    def test_cli_args_reach_longvie_sampling_params(self):
        args = self._parse(
            ["--longvie-dense-video", "depth.mp4", "--longvie-sparse-video", "t.mp4"]
        )
        cli_args = LongVie2SamplingParams.get_cli_args(args)
        self.assertEqual(cli_args["longvie_dense_video"], "depth.mp4")
        self.assertEqual(cli_args["longvie_sparse_video"], "t.mp4")

    def test_cli_args_are_ignored_by_other_wan_models(self):
        args = self._parse(["--longvie-dense-video", "depth.mp4"])
        self.assertNotIn(
            "longvie_dense_video", WanI2V_14B_480P_SamplingParam.get_cli_args(args)
        )

    def test_control_paths_reach_the_request(self):
        params = LongVie2SamplingParams(
            prompt="a video",
            longvie_dense_video="depth.mp4",
            longvie_sparse_video="track.mp4",
        )
        req = Req(sampling_params=params)
        # the Req fields shadow the sampling params until they are applied
        self.assertIsNone(req.longvie_dense_video)

        params.apply_request_extra(req)
        self.assertEqual(req.longvie_dense_video, "depth.mp4")
        self.assertEqual(req.longvie_sparse_video, "track.mp4")

    def test_registry_resolves_longvie_sampling_params(self):
        model_info = get_model_info("Vchitect/LongVie2")
        self.assertIs(model_info.sampling_param_cls, LongVie2SamplingParams)


if __name__ == "__main__":
    unittest.main()
