#!/usr/bin/env python3
"""Client for the resident surrogate service.

``ask_surrogate(model, image_paths, question)`` submits one job and polls for
its terminal file.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from impedit.config import get_paths  # noqa: E402

PATHS = get_paths()
_queue_root = PATHS.tmp / "queues"
QUEUE = str(_queue_root / "surrogate_queue")
DONE = str(_queue_root / "surrogate_done")
FAILED = str(_queue_root / "surrogate_failed")


def ask_surrogate(model, image_paths, question, timeout=300):
    req_id = uuid.uuid4().hex
    job = {"request_id": req_id, "model": model,
           "imgs": [os.path.abspath(p) for p in image_paths],
           "question": question}
    os.makedirs(QUEUE, exist_ok=True)
    tmp = os.path.join(QUEUE, req_id + ".tmp")
    with open(tmp, "w") as f:
        json.dump(job, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(QUEUE, req_id + ".json"))
    start = time.time()
    while time.time() - start < timeout:
        done_p = os.path.join(DONE, req_id + ".json")
        fail_p = os.path.join(FAILED, req_id + ".json")
        if os.path.exists(done_p):
            try:
                with open(done_p) as f:
                    r = json.load(f)
            except (ValueError, OSError):
                time.sleep(0.02)
                continue
            os.remove(done_p)
            return r.get("answer")
        if os.path.exists(fail_p):
            try:
                with open(fail_p) as f:
                    r = json.load(f)
            except (ValueError, OSError):
                time.sleep(0.02)
                continue
            os.remove(fail_p)
            return {"ERR": r.get("error", "ERR"), "status": r.get("status", "ERR")}
        time.sleep(0.02)
    return {"ERR": "TIMEOUT"}


def is_not_loaded(res):
    return isinstance(res, dict) and res.get("status") == "ERR_MODEL_NOT_LOADED"
