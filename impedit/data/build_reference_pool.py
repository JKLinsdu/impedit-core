"""Render the attacker's own reference pool: one stand-in image per target query.

The attack never reads the deployment knowledge base.  The output is a JSON list
of image paths, consumed as ``REF_POOL`` when training with ``KB_FREE=1``.

    python -m impedit.data.build_reference_pool --task webqa
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from impedit.config import get_paths

PROMPT = "A photograph that shows the answer to the question: %s"

BASES = {
    "webqa": {"rid": "gtproxy", "idx": 950000, "rnd": 2000, "seed": 9000, "tag": "gt",
              "out": "ref_pool_gt_webqa.json", "timeout": 2400},
    "mmqa": {"rid": "gtmmqa", "idx": 960000, "rnd": 3000, "seed": 11000, "tag": "gtm",
             "out": "ref_pool_gt_mmqa.json", "timeout": 3000},
}


def default_poison_json(task: str) -> Path:
    return get_paths().work / "poisoned" / ("poison_%s.json" % task)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(BASES), required=True)
    ap.add_argument("--poison-json", default=None,
                    help="target manifest (default: $IMPEDIT_WORK/poisoned/poison_<task>.json)")
    ap.add_argument("--out", default=None,
                    help="output list of image paths (default: $IMPEDIT_WORK/<task pool>)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="seconds to wait for the renderer (default: %(default)s)")
    args = ap.parse_args()

    base = BASES[args.task]
    paths = get_paths()
    queue_root = paths.tmp / "queues"
    QUEUE, DONE, FAILED = (queue_root / "render_queue", queue_root / "render_done",
                           queue_root / "render_failed")
    for d in (QUEUE, DONE, FAILED):
        d.mkdir(parents=True, exist_ok=True)

    src = Path(args.poison_json) if args.poison_json else default_poison_json(args.task)
    if not src.exists():
        raise SystemExit("target manifest not found: %s\n"
                         "pass --poison-json (MMQA's manifest is built by "
                         "`python -m impedit.data.build_poison_manifest mmqa`)" % src)
    out = Path(args.out) if args.out else paths.work / base["out"]
    timeout = args.timeout if args.timeout is not None else base["timeout"]

    recs = json.load(open(src, encoding="utf-8"))
    print("manifest %s: %d targets" % (src, len(recs)), flush=True)

    ids = []
    for i, rec in enumerate(recs):
        rid = "%s_%05d" % (base["rid"], i)
        job = {"request_id": rid,
               "idx": base["idx"] + i,
               "rnd": base["rnd"] + i,
               "seed": base["seed"] + i,
               "cap": PROMPT % rec["question"],
               "tag": base["tag"]}
        tmp = QUEUE / (rid + ".json.tmp")
        tmp.write_text(json.dumps(job), encoding="utf-8")
        os.replace(tmp, QUEUE / (rid + ".json"))
        ids.append(rid)
    print("submitted %d render requests" % len(ids), flush=True)

    t0, got, missing = time.time(), [], set(ids)
    while missing and time.time() - t0 < timeout:
        for rid in list(missing):
            done, failed = DONE / (rid + ".json"), FAILED / (rid + ".json")
            if done.exists():
                got.append(json.loads(done.read_text(encoding="utf-8"))["image_path"])
                missing.discard(rid)
            elif failed.exists():
                err = json.loads(failed.read_text(encoding="utf-8")).get("error", "")
                print("FAILED %s %s" % (rid, str(err)[:80]), flush=True)
                missing.discard(rid)
        time.sleep(2)

    out.write_text(json.dumps(sorted(got)), encoding="utf-8")
    print("wrote %s with %d images (missing %d) in %.0fs"
          % (out, len(got), len(missing), time.time() - t0), flush=True)
    return 0 if got and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
