#!/usr/bin/env python3
"""Resident surrogate MLLM service on a directory queue."""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PAYLOAD_DIR = str(Path(__file__).resolve().parent)
if _PAYLOAD_DIR not in sys.path:
    sys.path.insert(0, _PAYLOAD_DIR)

from impedit.config import apply_runtime_env, get_paths  # noqa: E402

PATHS = get_paths()
apply_runtime_env(PATHS)

_ap = argparse.ArgumentParser(description="Resident surrogate MLLM service.")
_ap.add_argument("--model", default=os.environ.get("SURROGATE_MODEL", "qwen"),
                 help="llava | qwen | qwen2vl | internvl3 | idefics3")
_ap.add_argument("--gpu", type=int, default=int(os.environ.get("SURROGATE_GPU", "0")))
_args = _ap.parse_args()

SURROGATE_MODEL = _args.model
SURROGATE_GPU = _args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(SURROGATE_GPU)

import torch  # noqa: E402,F401

try:
    from . import surrogate_mllm  # noqa: E402
except ImportError:  # direct script execution
    import surrogate_mllm  # noqa: E402

_queue_root = PATHS.tmp / "queues"
QUEUE = _queue_root / "surrogate_queue"
RUNNING = _queue_root / "surrogate_running"
DONE = _queue_root / "surrogate_done"
FAILED = _queue_root / "surrogate_failed"
for d in (QUEUE, RUNNING, DONE, FAILED):
    d.mkdir(parents=True, exist_ok=True)

PATHS.logs.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=str(PATHS.logs / ("vlm_%s.log" % SURROGATE_MODEL)),
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

print("Loading VLM %s (persistent) on cuda:%d..." % (SURROGATE_MODEL, SURROGATE_GPU), flush=True)
m, proc = surrogate_mllm.load_surrogate(SURROGATE_MODEL, device="cuda:0")
print("loaded %s" % SURROGATE_MODEL, flush=True)


def write_terminal(kind, req_id, payload):
    dir_ = DONE if kind == "done" else FAILED
    p = os.path.join(str(dir_), req_id + ".json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


while True:
    jobs = sorted(glob.glob(os.path.join(str(QUEUE), "*.json")))
    if not jobs:
        time.sleep(0.02)
        continue
    job_path = None
    for cand in jobs:
        try:
            with open(cand) as f:
                peek = json.load(f)
        except Exception:
            continue
        if peek.get("model") == SURROGATE_MODEL:
            job_path = cand
            break
    if job_path is None:
        time.sleep(0.02)
        continue
    req_id = os.path.splitext(os.path.basename(job_path))[0]
    try:
        os.replace(job_path, os.path.join(str(RUNNING), req_id + ".json"))
    except FileNotFoundError:
        continue
    try:
        with open(os.path.join(str(RUNNING), req_id + ".json")) as f:
            job = json.load(f)
        model = job["model"]
        imgs = job["imgs"]
        question = job["question"]
        request_id = job["request_id"]
    except Exception:
        logging.exception("CRASH bad-job request=%s", req_id)
        write_terminal("failed", req_id, {"request_id": req_id, "status": "ERR_BADJOB"})
        os.remove(os.path.join(str(RUNNING), req_id + ".json"))
        continue

    if model != SURROGATE_MODEL:
        write_terminal("failed", req_id, {"request_id": request_id, "model": model,
                                          "status": "ERR_MODEL_NOT_LOADED",
                                          "loaded_model": SURROGATE_MODEL})
        os.remove(os.path.join(str(RUNNING), req_id + ".json"))
        continue

    try:
        ans = surrogate_mllm.surrogate_mllm(m, proc, imgs, question, SURROGATE_MODEL)
        write_terminal("done", req_id, {"request_id": request_id, "model": model,
                                        "answer": ans})
        logging.info("OK %s %s", model, str(ans)[:60])
    except Exception as e:
        logging.exception("FAIL request=%s err=%s", req_id, str(e)[:120])
        write_terminal("failed", req_id, {"request_id": request_id, "model": model,
                                          "status": "ERR", "error": str(e)[:200]})
    os.remove(os.path.join(str(RUNNING), req_id + ".json"))
