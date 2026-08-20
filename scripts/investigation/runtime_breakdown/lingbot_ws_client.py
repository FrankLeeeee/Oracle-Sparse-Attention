# SPDX-License-Identifier: Apache-2.0
"""Drive one LingBot-World realtime WebSocket session and record timings.

Sends a single init payload (condition image + prompt + num_frames); the
server then streams every chunk of the session. Records per-chunk wall-clock
arrival times plus the server-reported chunk_stats (scheduler_forward_ms is
the full per-chunk pipeline forward), and writes them to a JSON file for
analyze.py.

    python lingbot_ws_client.py --port 32002 --model-path robbyant/... \
        --prompt "..." --first-frame frame.png --size 832x480 \
        --num-frames 165 --out session.json [--profile] [--save-video out.mp4]
"""

import argparse
import asyncio
import json
import pathlib
import time

import msgspec.msgpack
import numpy as np
import websockets

RECV_TIMEOUT_S = 900.0


async def run_session(args: argparse.Namespace) -> dict:
    latent_frames = (args.num_frames - 1) // 4 + 1
    expected_chunks = (latent_frames + 2) // 3
    init_payload = {
        "type": "init",
        "model": args.model_path,
        "prompt": args.prompt,
        "size": args.size,
        "num_frames": args.num_frames,
        "seed": 42,
        "first_frame": pathlib.Path(args.first_frame).read_bytes(),
        "realtime_output_format": "raw",
    }
    if args.profile:
        init_payload["profile"] = True
        init_payload["num_profiled_timesteps"] = args.num_profiled_timesteps

    url = f"ws://127.0.0.1:{args.port}/v1/realtime_video/generate"
    chunk_events: list[dict] = []
    chunk_stats: list[dict] = []
    # Frames are only materialized when a video is asked for: a 20s 720p
    # session is ~1.7 GB of raw RGB.
    frames: list[np.ndarray] | None = [] if args.save_video else None
    frames_received = 0
    session_start = time.time()
    first_chunk_at = None
    async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
        await ws.send(msgspec.msgpack.encode(init_payload))
        received_chunks: set[int] = set()
        while len(received_chunks) < expected_chunks:
            try:
                header_payload = await asyncio.wait_for(
                    ws.recv(), timeout=RECV_TIMEOUT_S
                )
            except (asyncio.TimeoutError, websockets.ConnectionClosed) as error:
                print(f"stream ended early: {error!r}")
                break
            header = msgspec.msgpack.decode(header_payload)
            message_type = header.get("type")
            if message_type == "error":
                raise RuntimeError(f"realtime generation failed: {header}")
            if message_type == "chunk_stats":
                chunk_stats.append({k: v for k, v in header.items() if k != "type"})
                continue
            if message_type == "frame_batch":
                header.pop("payload", None)
            elif message_type == "frame_batch_header":
                payload = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
                if frames is not None:
                    frames.extend(_decode_raw_rgb(header, payload))
            else:
                raise ValueError(f"unexpected realtime message: {header}")
            now = time.time()
            if first_chunk_at is None:
                first_chunk_at = now
            frames_received += int(header.get("num_frames", 0))
            chunk_index = int(header["chunk_index"])
            if header.get("is_final_frame_batch", True):
                received_chunks.add(chunk_index)
                chunk_events.append({"chunk_index": chunk_index, "wall_time": now})
    session_end = time.time()
    if frames:
        _write_video(frames, args.save_video, fps=args.fps)
    return {
        "num_frames_requested": args.num_frames,
        "expected_chunks": expected_chunks,
        "received_chunks": len(chunk_events),
        "frames_received": frames_received,
        "session_wall_s": session_end - session_start,
        "time_to_first_chunk_s": (
            (first_chunk_at - session_start) if first_chunk_at else None
        ),
        "stream_wall_s": ((session_end - first_chunk_at) if first_chunk_at else None),
        "session_start": session_start,
        "session_end": session_end,
        "chunk_events": chunk_events,
        "chunk_stats": chunk_stats,
        "profile": bool(args.profile),
    }


def _decode_raw_rgb(header: dict, payload: bytes) -> list[np.ndarray]:
    """Split one raw-RGB frame batch into per-frame HxWx3 uint8 arrays."""
    height, width = int(header["height"]), int(header["width"])
    channels = int(header.get("channels", 3))
    array = np.frombuffer(payload, dtype=np.uint8)
    expected = int(header["num_frames"]) * height * width * channels
    if array.size != expected:
        raise ValueError(
            f"frame batch payload size mismatch: {array.size} != {expected}"
        )
    return list(array.reshape(-1, height, width, channels))


def _write_video(frames: list[np.ndarray], path: str, *, fps: int) -> None:
    import imageio.v2 as imageio

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"wrote {len(frames)} frames -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--first-frame", required=True)
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num-profiled-timesteps", type=int, default=40)
    parser.add_argument(
        "--save-video",
        default=None,
        help="write the streamed frames to this mp4 (off by default)",
    )
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    result = asyncio.run(run_session(args))
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    print(
        f"session done: {result['received_chunks']}/{result['expected_chunks']} "
        f"chunks, {result['frames_received']} frames in "
        f"{result['session_wall_s']:.1f}s -> {args.out}"
    )


if __name__ == "__main__":
    main()
