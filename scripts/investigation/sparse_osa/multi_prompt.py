# SPDX-License-Identifier: Apache-2.0
"""Multi-prompt validation: 5 new prompts x all methods at the ~0.30 tier.

Generates one 720p / 20 s video per (prompt, method) with the same calibrated
configs as the main sweep, then renders one frame-comparison sheet per prompt
(labeled rows: dense + methods; labeled columns: 7 frames from t = 1 to 19 s).

    python multi_prompt.py [--gpus 0,1,6] [--sheets-only]
Videos -> runs_prompts/<prompt>_<method>/, sheets -> prompt_sheet_<p>.png,
timings/densities -> results_prompts.json.
"""

import argparse
import json
import pathlib
import queue
import sys
import threading

from common import METHOD_LABELS, extract_frames, render_frame_sheet, run_generate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("sparse_osa")
TARGET = "0.3"

PROMPTS = {
    "p1_forest": (
        "A dynamic and chaotic scene in a dense forest during a heavy rainstorm, "
        "capturing a real girl frantically running through the foliage. Her wild "
        "hair flows behind her as she sprints, her arms flailing and her face "
        "contorted in fear and desperation. Behind her, various animals—rabbits, "
        "deer, and birds—are also running, creating a frenzied atmosphere. The "
        "girl's clothes are soaked, clinging to her body, and she is screaming "
        "and shouting as she tries to escape. The background is a blur of "
        "greenery and rain-drenched trees, with occasional glimpses of the "
        "darkening sky. A wide-angle shot from a low angle, emphasizing the "
        "urgency and chaos of the moment."
    ),
    "p2_plating": (
        "A dynamic over-the-shoulder perspective of a chef meticulously plating "
        "a dish in a bustling kitchen. The chef, a middle-aged man with a neatly "
        "trimmed beard and focused expression, deftly arranges ingredients on a "
        "pristine white plate. His hands move with precision, each gesture "
        "deliberate and practiced. The background shows a crowded kitchen with "
        "steaming pots, whirring blenders, and the clatter of utensils. Bright "
        "lights highlight the scene, casting shadows across the busy workspace. "
        "The camera angle captures the chef's detailed work from behind, "
        "emphasizing his skill and dedication."
    ),
    "p3_raccoon": (
        "A playful raccoon is seen playing an electronic guitar, strumming the "
        "strings with its front paws. The raccoon has distinctive black facial "
        "markings and a bushy tail. It sits comfortably on a small stool, its "
        "body slightly tilted as it focuses intently on the instrument. The "
        "setting is a cozy, dimly lit room with vintage posters on the walls, "
        "adding a retro vibe. The raccoon's expressive eyes convey a sense of "
        "joy and concentration. Medium close-up shot, focusing on the raccoon's "
        "face and hands interacting with the guitar."
    ),
    "p4_teacup": (
        "A close-up shot of a ceramic teacup slowly pouring water into a glass "
        "mug. The water flows smoothly from the spout of the teacup into the "
        "mug, creating gentle ripples as it fills up. Both cups have detailed "
        "textures, with the teacup having a matte finish and the glass mug "
        "showcasing clear transparency. The background is a blurred kitchen "
        "countertop, adding context without distracting from the central "
        "action. The pouring motion is fluid and natural, emphasizing the "
        "interaction between the two cups."
    ),
    "p5_tsunami": (
        "A dramatic and dynamic scene in the style of a disaster movie, "
        "depicting a powerful tsunami rushing through a narrow alley in "
        "Bulgaria. The water is turbulent and chaotic, with waves crashing "
        "violently against the walls and buildings on either side. The alley "
        "is lined with old, weathered houses, their facades partially "
        "submerged and splintered. The camera angle is low, capturing the full "
        "force of the tsunami as it surges forward, creating a sense of "
        "urgency and danger. People can be seen running frantically, adding to "
        "the chaos. The background features a distant horizon, hinting at the "
        "larger scale of the tsunami. A dynamic, sweeping shot from a "
        "low-angle perspective, emphasizing the movement and intensity of the "
        "event."
    ),
}


def method_list() -> list[tuple[str, str | None, dict | None]]:
    configs = json.loads((ROOT / "configs.json").read_text())
    methods: list[tuple[str, str | None, dict | None]] = [("dense", None, None)]
    for method in ("osa", "lightforcing", "radial", "svg1", "svg2", "xattention"):
        entry = configs.get(method, {}).get(TARGET)
        if entry:
            methods.append((method, method, entry["config"]))
    return methods


def generate_all(gpus: list[int]) -> None:
    jobs: queue.Queue = queue.Queue()
    for prompt_id, prompt in PROMPTS.items():
        for tag, method, config in method_list():
            jobs.put((prompt_id, prompt, tag, method, config))
    results_path = ROOT / "results_prompts.json"
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    lock = threading.Lock()

    def worker(index: int, gpu: int) -> None:
        port_base = 41000 + index * 20
        while True:
            try:
                prompt_id, prompt, tag, method, config = jobs.get_nowait()
            except queue.Empty:
                return
            run_id = f"{prompt_id}_{tag}"
            try:
                result = run_generate(
                    out_dir=ROOT / "runs_prompts" / run_id,
                    log_name="timing.log",
                    gpu=gpu,
                    port_base=port_base,
                    width=1280,
                    height=720,
                    num_frames=321,
                    method=method,
                    method_config=config,
                    save_output=True,
                    timeout_s=2400,
                    prompt=prompt,
                )
                with lock:
                    results[run_id] = result
                    results_path.write_text(json.dumps(results, indent=2))
                print(
                    f"[gpu{gpu}] DONE {run_id} rc={result['returncode']} "
                    f"denoise={result.get('denoise_s')} density={result.get('density')}",
                    flush=True,
                )
            except Exception as error:
                print(f"[gpu{gpu}] FAIL {run_id}: {error}", flush=True)
            finally:
                jobs.task_done()

    threads = [
        threading.Thread(target=worker, args=(i, g), daemon=True)
        for i, g in enumerate(gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("GENERATION DONE", flush=True)


def build_sheets() -> None:
    fps = 16
    frame_indices = [second * fps for second in (1, 4, 7, 10, 13, 16, 19)]
    for prompt_id in PROMPTS:
        rows = []
        for tag, _, _ in method_list():
            frames = extract_frames(
                ROOT / "runs_prompts" / f"{prompt_id}_{tag}", frame_indices
            )
            if frames is None:
                print(f"missing video: {prompt_id}_{tag}", flush=True)
                continue
            rows.append((METHOD_LABELS[tag], frames))
        if not rows:
            continue
        out = ROOT / f"prompt_sheet_{prompt_id}.png"
        render_frame_sheet(
            rows=rows, frame_indices=frame_indices, fps=fps, out_path=out
        )
        print(f"wrote {out} rows={[label for label, _ in rows]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,6")
    parser.add_argument("--sheets-only", action="store_true")
    args = parser.parse_args()
    if not args.sheets_only:
        generate_all([int(g) for g in args.gpus.split(",")])
    build_sheets()


if __name__ == "__main__":
    main()
