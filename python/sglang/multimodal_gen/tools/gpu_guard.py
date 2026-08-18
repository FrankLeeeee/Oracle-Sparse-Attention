# SPDX-License-Identifier: Apache-2.0
"""Refuse to report a timing that another process could have influenced.

This box has eight GPUs shared with other people's jobs, and a neighbour at 86%
utilization silently doubled a dense-attention measurement during this work
(1.39 ms read as 2.4-3.0 ms). Numbers taken that way are worse than no numbers,
because they look plausible.

So every benchmark here runs inside :func:`exclusive_gpu`, which

1. refuses to start unless the target device has no other compute process,
2. samples the device's process list in a background thread while the
   measurement runs, and
3. raises :class:`GpuInterference` afterwards if any foreign PID appeared.

Usage::

    with exclusive_gpu() as gpu:          # picks an idle device, or
    with exclusive_gpu(device=7) as gpu:  # verifies the one you name
        ...                               # measure here
    print(gpu.samples)                    # utilization trace, for the record

The check is deliberately conservative: it counts *processes*, not utilization,
because a process that is idle at sampling time can wake up mid-measurement.
"""

import contextlib
import os
import subprocess
import threading
import time

import msgspec


class GpuInterference(RuntimeError):
    """Another process was on the GPU before, during, or after a measurement."""


class GpuSample(msgspec.Struct, frozen=True):
    seconds: float
    utilization_percent: int
    memory_mib: int
    foreign_pids: tuple[int, ...]


class GpuSession(msgspec.Struct):
    """The device a measurement ran on, plus the occupancy trace it saw."""

    index: int
    uuid: str
    name: str
    samples: list[GpuSample] = []

    @property
    def peak_foreign_memory_mib(self) -> int:
        return max((s.memory_mib for s in self.samples if s.foreign_pids), default=0)


def _nvidia_smi(*query: str) -> list[list[str]]:
    output = subprocess.run(
        ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not output:
        return []
    return [[cell.strip() for cell in line.split(",")] for line in output.splitlines()]


def _devices() -> list[tuple[int, str, str]]:
    return [
        (int(index), uuid, name)
        for index, uuid, name in _nvidia_smi("--query-gpu=index,uuid,name")
    ]


def _compute_apps() -> dict[str, list[tuple[int, int]]]:
    """``{gpu_uuid: [(pid, used_mib), ...]}`` for every running compute process."""
    apps: dict[str, list[tuple[int, int]]] = {}
    for uuid, pid, memory in _nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,used_memory"
    ):
        apps.setdefault(uuid, []).append((int(pid), int(memory)))
    return apps


def _foreign_pids(uuid: str, own_pids: set[int]) -> tuple[int, ...]:
    return tuple(
        pid for pid, _ in _compute_apps().get(uuid, []) if pid not in own_pids
    )


def _utilization(index: int) -> tuple[int, int]:
    rows = _nvidia_smi(f"--id={index}", "--query-gpu=utilization.gpu,memory.used")
    return (int(rows[0][0]), int(rows[0][1])) if rows else (0, 0)


def find_idle_gpu(*, exclude: frozenset[int] = frozenset()) -> tuple[int, str, str]:
    """Lowest-numbered device with no foreign compute process on it."""
    own = {os.getpid()}
    apps = _compute_apps()
    for index, uuid, name in _devices():
        if index in exclude:
            continue
        if not [pid for pid, _ in apps.get(uuid, []) if pid not in own]:
            return index, uuid, name
    raise GpuInterference(
        "every GPU has another compute process on it; nothing here can be timed"
    )


@contextlib.contextmanager
def exclusive_gpu(
    *,
    device: int | None = None,
    poll_seconds: float = 1.0,
    own_pids: set[int] | None = None,
):
    """Run a measurement on a GPU nobody else is using, or fail loudly.

    ``device=None`` picks an idle one and exports ``CUDA_VISIBLE_DEVICES`` so
    the caller's later CUDA init lands there — so enter this *before* touching
    torch.cuda. Naming a device instead only verifies it.
    """
    own = set(own_pids or set()) | {os.getpid()}
    if device is None:
        index, uuid, name = find_idle_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    else:
        matches = [row for row in _devices() if row[0] == device]
        if not matches:
            raise GpuInterference(f"no such GPU: {device}")
        index, uuid, name = matches[0]
        intruders = _foreign_pids(uuid, own)
        if intruders:
            raise GpuInterference(
                f"GPU {index} already has foreign process(es) {intruders}; "
                "pick another device or wait"
            )

    session = GpuSession(index=index, uuid=uuid, name=name)
    stop = threading.Event()
    started = time.perf_counter()

    def poll() -> None:
        while not stop.is_set():
            utilization, memory = _utilization(index)
            session.samples.append(
                GpuSample(
                    seconds=round(time.perf_counter() - started, 2),
                    utilization_percent=utilization,
                    memory_mib=memory,
                    foreign_pids=_foreign_pids(uuid, own),
                )
            )
            stop.wait(poll_seconds)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    try:
        yield session
    finally:
        stop.set()
        watcher.join(timeout=5.0)

    intruders = sorted({pid for s in session.samples for pid in s.foreign_pids})
    if intruders:
        raise GpuInterference(
            f"GPU {index} was shared with process(es) {intruders} during the "
            f"measurement (peak foreign memory {session.peak_foreign_memory_mib} "
            "MiB); discard these numbers"
        )


def main() -> None:
    """``python -m sglang.multimodal_gen.tools.gpu_guard`` — report idle devices."""
    own = {os.getpid()}
    apps = _compute_apps()
    for index, uuid, name in _devices():
        foreign = [(pid, mem) for pid, mem in apps.get(uuid, []) if pid not in own]
        utilization, memory = _utilization(index)
        state = "IDLE" if not foreign else f"BUSY {foreign}"
        print(f"GPU {index}  {name}  {utilization:3d}%  {memory:6d} MiB  {state}")


if __name__ == "__main__":
    main()
