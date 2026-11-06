#!/usr/bin/env python3
"""Assemble the poison manifest read by the evaluator.

Captions in the unified index are the draft used for the first render; only the
payload loop's manifest carries the final text.

    python -m impedit.data.build_poison_manifest mmqa --surrogate qwen
    python -m impedit.data.build_poison_manifest webqa --source WebQA-100-strat-regen.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from impedit.config import get_paths
from impedit.data import layout

DEFAULT_SURROGATES = ["qwen", "llava", "qwen2vl", "internvl3", "idefics3"]


def _check(recs, out):
    """A manifest must cover exactly the target set, with no duplicate qid."""
    want = int(os.environ.get("TARGETS", "100"))
    qids = [r["qid"] for r in recs]
    if len(recs) != want:
        raise SystemExit("ERROR: %s has %d records, expected %d (partial payloads are not "
                         "silently accepted)" % (out, len(recs), want))
    if len(set(qids)) != len(qids):
        raise SystemExit("ERROR: %s has duplicate qids" % out)


def build_mmqa(surr: str, index_path, images_tmpl: str, out_path) -> int:
    targets = [r for r in json.load(open(index_path, encoding="utf-8"))
               if r.get("role") == "target"]
    imgdir = images_tmpl.format(surr=surr)
    recs, missing = [], []
    for i, r in enumerate(targets):
        p = os.path.join(imgdir, "q%d.png" % i)
        if not os.path.exists(p):
            missing.append(i)
            continue
        recs.append({
            "qid": r["qid"],
            "question": r["question"],
            "wrong_answer": r["wrong_answer"],
            "gt_answer": r["gt_answer"],
            "poisoned_img_path": p,
            "poisoned_caption": r.get("poisoned_caption", ""),
            "qcate": r.get("Qcate", r.get("qcate", "")),
        })
    _check(recs, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("%-10s n=%-4d missing %-3d %s -> %s"
          % (surr, len(recs), len(missing), missing[:6], out_path))
    return 0 if recs else 1


def cmd_mmqa(a) -> int:
    paths = get_paths()
    index_path = a.index or layout.mmqa_index_json(paths)
    images_tmpl = a.images_dir or str(paths.poisoned / "mmqa_surrogate_{surr}_100")
    surrs = a.surrogate or DEFAULT_SURROGATES
    rc = 0
    for surr in surrs:
        if a.out:
            out = a.out.format(surr=surr)
        elif len(surrs) == 1:
            out = str(layout.poison_manifest(paths, "mmqa"))
        else:
            out = str(paths.poisoned / ("poison_mmqa_%s.json" % surr))
        rc |= build_mmqa(surr, index_path, images_tmpl, Path(out))
    return rc


def cmd_webqa(a) -> int:
    paths = get_paths()
    src = a.source or layout.webqa_targets_json(paths)
    out = a.out or str(layout.poison_manifest(paths, "webqa"))

    rows = json.load(open(src, encoding="utf-8"))
    recs, missing = [], []
    for r in rows:
        p = r.get("poisoned_img_path", "")
        if not os.path.exists(p):
            missing.append(r.get("qid"))
            if not a.keep_missing:
                continue
        recs.append({
            "qid": r["qid"],
            "question": r["question"],
            "wrong_answer": r["wrong_answer"],
            "gt_answer": r["gt_answer"],
            "poisoned_img_path": p,
            "poisoned_caption": r.get("poisoned_caption", ""),
            "qcate": r.get("Qcate", r.get("qcate", "")),
        })

    if not recs:
        raise SystemExit("no poisoned images found; check %s" % src)
    _check(recs, out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("webqa n=%d  missing=%d %s -> %s" % (len(recs), len(missing), missing[:6], out))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("mmqa", help="unified index + q{i}.png -> poison_mmqa.json")
    p1.add_argument("--surrogate", nargs="*", default=None,
                    help="one or more surrogate names (default: all five)")
    p1.add_argument("--index", default=None)
    p1.add_argument("--images-dir", default=None,
                    help="directory template containing {surr}; default IMPEDIT_WORK/poisoned/mmqa_surrogate_{surr}_100")
    p1.add_argument("--out", default=None, help="output path template containing {surr}")
    p1.set_defaults(func=cmd_mmqa)

    p2 = sub.add_parser("webqa", help="WebQA target file -> poison_webqa.json")
    p2.add_argument("--source", default=None)
    p2.add_argument("--out", default=None)
    p2.add_argument("--keep-missing", action="store_true",
                    help="keep records whose image is absent (default: drop them)")
    p2.set_defaults(func=cmd_webqa)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
