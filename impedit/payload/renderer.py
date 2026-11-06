"""Resident SD3.5 rendering service built on a directory queue.

A client writes ``render_queue/<id>.json``; the server claims it with
``os.replace`` and leaves exactly one of ``render_done/<id>.json`` or
``render_failed/<id>.json``.
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from impedit.config import apply_runtime_env, get_paths  # noqa: E402

PATHS = get_paths()
apply_runtime_env(PATHS)

_ap = argparse.ArgumentParser(description="Resident SD3.5 rendering service.")
_ap.add_argument("--gpu", default=os.environ.get("RENDER_GPU", "0"))
_ap.add_argument("--offload", action="store_true",
                 default=os.environ.get("RENDER_OFFLOAD", "0") == "1")
_args = _ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import torch  # noqa: E402
from diffusers import StableDiffusion3Pipeline  # noqa: E402

MODEL_DIR = PATHS.models / "SD3.5-M"

_queue_root = PATHS.tmp / "queues"
QUEUE = _queue_root / "render_queue"
RUNNING = _queue_root / "render_running"
DONE = _queue_root / "render_done"
FAILED = _queue_root / "render_failed"
OUT = Path(os.environ.get("RENDER_OUT") or (PATHS.work / "render_cache"))
for d in (QUEUE, RUNNING, DONE, FAILED, OUT):
    d.mkdir(parents=True, exist_ok=True)

PATHS.logs.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=str(PATHS.logs / "render_server.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

print("Loading SD3.5-M (persistent)...", flush=True)
if _args.offload:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    print("cpu offload enabled", flush=True)
else:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16, device_map="cuda"
    )
pipe.set_progress_bar_config(disable=True)
print("loaded", flush=True)

RETRYABLE = ("CUDA out of memory", "OutOfMemoryError", "CUDA error", "device-side assert",
             "execution timeout", "[enforce fail]", "Connection timed out", "timed out",
             "run-time error")
MAX_ATTEMPTS = 2


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_terminal(kind, req_id, payload):
    """kind='done'|'failed'; exactly one terminal file per request."""
    dir_ = DONE if kind == "done" else FAILED
    with open(os.path.join(str(dir_), req_id + ".json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False)


while True:
    jobs = sorted(glob.glob(os.path.join(str(QUEUE), "*.json")))
    if not jobs:
        time.sleep(0.2)
        continue
    job_path = jobs[0]
    req_id = os.path.splitext(os.path.basename(job_path))[0]
    try:
        os.replace(job_path, os.path.join(str(RUNNING), req_id + ".json"))
    except FileNotFoundError:
        continue
    try:
        with open(os.path.join(str(RUNNING), req_id + ".json")) as f:
            job = json.load(f)
        idx = job["idx"]
        cap = job["cap"]
        rnd = job["rnd"]
        seed = job.get("seed", 42 + rnd)
        tag = str(job.get("tag", "") or "")
        request_id = job["request_id"]
    except Exception:
        logging.exception("CRASH bad-job request=%s", req_id)
        write_terminal("failed", req_id, {"request_id": req_id, "status": "ERR_BADJOB",
                                          "error": "bad json / missing key (not retryable)"})
        try:
            os.remove(os.path.join(str(RUNNING), req_id + ".json"))
        except FileNotFoundError:
            pass
        continue

    attempts = 0
    status = "ERR"
    msg = ""
    img_path = ""
    t0 = time.time()
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            gen = torch.Generator().manual_seed(seed)
            t0 = time.time()
            img = pipe(prompt=cap, generator=gen, num_inference_steps=28,
                       guidance_scale=3.5).images[0]
            img_path = os.path.join(str(OUT), "idx%d%s_r%d.png" % (
                idx, ("_" + tag) if tag else "", rnd))
            img.save(img_path)
            status = "ok"
            break
        except Exception as e:
            msg = str(e)[:200]
            if not any(r in msg for r in RETRYABLE) or attempts >= MAX_ATTEMPTS:
                break
            logging.warning("retry request=%s idx=%d rnd=%d attempts=%d err=%s",
                            req_id, idx, rnd, attempts, msg[:80])
            time.sleep(1)

    if status == "ok":
        m = {"request_id": request_id, "idx": idx, "rnd": rnd, "seed": seed,
             "status": "ok", "tag": tag,
             "image_path": img_path, "image_sha256": sha256_file(img_path),
             "render_seconds": round(time.time() - t0, 2), "attempts": attempts,
             "cap": cap,
             "cap_sha256": hashlib.sha256(cap.encode()).hexdigest(),
             "prompt_sha256": hashlib.sha256(cap.encode()).hexdigest()}
        write_terminal("done", req_id, m)
        logging.info("OK request=%s idx=%d rnd=%d tag=%s %.2fs",
                     req_id, idx, rnd, tag, m["render_seconds"])
    else:
        logging.error("FAIL request=%s idx=%d rnd=%d err=%s", req_id, idx, rnd, msg)
        write_terminal("failed", req_id, {"request_id": request_id, "idx": idx,
                                          "rnd": rnd, "seed": seed, "status": "ERR",
                                          "error": msg, "attempts": attempts})
    try:
        os.remove(os.path.join(str(RUNNING), req_id + ".json"))
    except FileNotFoundError:
        pass
