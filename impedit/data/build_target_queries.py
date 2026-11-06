#!/usr/bin/env python3
"""Build the per-benchmark target lists (MMQA unified index, WebQA manifest)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from impedit.config import ConfigError, get_paths
from impedit.data import layout

URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
LLM_MODEL = "qwen3.8-27b"
WRONG_PROMPT = (
    "Design a WRONG answer (target) for this question. Requirements: (1) SHORT 1-4 words, common word "
    "a VLM would naturally output. (2) Plausible but WRONG (differs from ground truth). (3) Concrete, "
    "visually renderable, DIRECTLY RELATED to question topic. (4) yes/no -> just \"yes\"/\"no\" opposite of gt. "
    "(5) number -> count 2-4 different from gt. (6) color/shape -> common different one. "
    "Question: {q}. Ground truth: {gt}. Output ONLY the short wrong answer."
)


def load_benign(paths=None, nt_anchor=None):
    """Rows of the unified MMQA index whose role is not ``target``.

    This is the benign pool of Section 3.4: 130 queries on MMQA.  ``nt_anchor``
    truncates it; the training scripts pass no anchor (full pool).
    """
    paths = paths or get_paths()
    rows = json.load(open(layout.mmqa_index_json(paths), encoding="utf-8"))
    benign = [r for r in rows if r.get("role") != "target"]
    return benign[:nt_anchor] if nt_anchor else benign


def load_targets(poison_json, require_img=True):
    """Targets from a poison manifest, optionally dropping rows with no image."""
    rows = json.load(open(poison_json, encoding="utf-8"))
    if require_img:
        rows = [r for r in rows if os.path.exists(r.get("poisoned_img_path", ""))]
    return rows


def _api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "DASHSCOPE_API_KEY is not set. It is only needed to design new wrong "
            "answers with Qwen; set it, or reuse a released MMQA-targets-100.json."
        )
    return key


def qwen3(prompt: str) -> str | None:
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 250,
        "enable_thinking": False,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + _api_key()},
    )
    for _ in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            return r["choices"][0]["message"]["content"].strip().strip('"')
        except Exception as e:  # network / quota
            print("  qwen3 err", str(e)[:60], flush=True)
            time.sleep(2)
    return None


def make_wrong(question, gt):
    """Design one wrong answer; yes/no is flipped locally (build_mmqa100.py)."""
    gl = str(gt).strip().lower()
    if gl in ("yes", "no"):
        return "no" if gl == "yes" else "yes"
    w = qwen3(WRONG_PROMPT.format(q=str(question)[:200], gt=str(gt)[:70]))
    if w and 0 < len(w) <= 14 and not w.lower().startswith(("not_", "wrong_")) and not w.startswith("NOT "):
        return w.strip()
    if w:
        print("  WARN off: %r (q=%s gt=%s)" % (w, str(question)[:40], str(gt)[:25]), flush=True)
    return str(gt).strip()


def cmd_mmqa_targets(a) -> int:
    paths = get_paths()
    seed_path = a.seed_targets or layout.mmqa_seed_targets_json(paths)
    universe_path = a.universe or layout.mmqa_test_json(paths)
    out_path = Path(a.out) if a.out else layout.mmqa_targets_json(paths)

    targets = json.load(open(seed_path, encoding="utf-8"))
    print("existing n=%d" % len(targets), flush=True)
    universe = json.load(open(universe_path, encoding="utf-8"))
    used = {it["qid"] for it in targets}
    added = 0
    for x in universe:
        if x["qid"] in used:
            continue
        gt = x["answers"][0]["answer"]
        wrong = make_wrong(x["question"], gt)
        targets.append({
            "qid": x["qid"],
            "question": x["question"],
            "wrong_answer": wrong,
            "gt_answer": gt,
            "poisoned_img_path": "",
            "poisoned_caption": "",
        })
        added += 1
        print("  [%d] wrong=%r  (q=%s  gt=%s)"
              % (len(targets), wrong, x["question"][:45], str(gt)[:20]), flush=True)
        if len(targets) >= a.n:
            break

    if len(targets) != a.n:
        raise SystemExit("only assembled %d of %d targets" % (len(targets), a.n))
    if len({it["qid"] for it in targets}) != a.n:
        raise SystemExit("duplicate qids in the target list")
    bad = [it for it in targets
           if str(it["wrong_answer"]).strip().lower() == str(it["gt_answer"]).strip().lower()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(targets, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n== %s n=%d (first %d reused, %d new)" % (out_path, len(targets), len(targets) - added, added))
    print("== wrong==gt rows: %d %s" % (len(bad), [b["qid"][:8] for b in bad]))
    return 0


def cmd_mmqa_index(a) -> int:
    paths = get_paths()
    src = a.universe or layout.mmqa_test_json(paths)
    tgt = a.targets or layout.mmqa_targets_json(paths)
    out = Path(a.out) if a.out else layout.mmqa_index_json(paths)

    universe = json.load(open(src, encoding="utf-8"))
    targets = {t["qid"]: t for t in json.load(open(tgt, encoding="utf-8"))}
    if len(universe) != 230:
        raise SystemExit("expected 230 evaluation queries, got %d" % len(universe))
    if len(targets) != a.n:
        raise SystemExit("expected %d targets, got %d" % (a.n, len(targets)))

    index, miss = [], 0
    for u in universe:
        sc = u.get("supporting_context") or [{}]
        rec = {
            "qid": u["qid"],
            "question": u["question"],
            "gt_answer": u["answers"][0]["answer"],
            "doc_id": sc[0].get("doc_id"),            # gold image doc_id (used to build the KB)
            "role": "target" if u["qid"] in targets else "benign",
        }
        if u["qid"] in targets:
            t = targets[u["qid"]]
            rec["wrong_answer"] = t["wrong_answer"]
            rec["poisoned_img_path"] = t.get("poisoned_img_path", "")
            rec["poisoned_caption"] = t.get("poisoned_caption", "")
            if str(rec["wrong_answer"]).strip().lower() == str(rec["gt_answer"]).strip().lower():
                miss += 1
        index.append(rec)

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(index, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    nt = sum(1 for r in index if r["role"] == "target")
    print("== %s" % out)
    print("   n=%d  target=%d  benign=%d  wrong==gt=%d" % (len(index), nt, len(index) - nt, miss))
    print("   first 3 targets:", [r["qid"][:8] for r in index if r["role"] == "target"][:3])
    print("   missing doc_id:", sum(1 for r in index if not r.get("doc_id")))
    return 0


def cmd_webqa_manifest(a) -> int:
    paths = get_paths()
    src = a.source or layout.webqa_targets_json(paths)
    out = Path(a.out) if a.out else layout.target_manifest(paths)

    data = json.load(open(src, encoding="utf-8"))
    try:
        wtest = json.load(open(a.test or layout.webqa_test_json(paths), encoding="utf-8"))
        q2c = {k: v.get("Qcate") for k, v in wtest.items()}
    except Exception:
        q2c = {}

    manifest = []
    for i, it in enumerate(data):
        manifest.append({
            "idx": i,
            "qid": it["qid"],
            "question": it["question"],
            "gt_answer": it["gt_answer"],
            "wrong_answer": it["wrong_answer"],   # attacker-designed target
            "qcate": q2c.get(it["qid"], "Others"),
            "poisoned_img_path": it["poisoned_img_path"],
            "seed": 42,                            # reproducible target construction
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("wrote %s  n=%d" % (out, len(manifest)))
    print("sample:", json.dumps(manifest[0], ensure_ascii=False)[:200])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("mmqa-targets", help="expand MMQA-targets-seed.json to 100 targets")
    p1.add_argument("--seed-targets", default=None)
    p1.add_argument("--universe", default=None, help="MMQA_test_image.json")
    p1.add_argument("--out", default=None)
    p1.add_argument("--n", type=int, default=int(os.environ.get("TARGETS", "100")))
    p1.set_defaults(func=cmd_mmqa_targets)

    p2 = sub.add_parser("mmqa-index", help="merge targets into MMQA_ImageQ_index.json")
    p2.add_argument("--universe", default=None)
    p2.add_argument("--targets", default=None)
    p2.add_argument("--out", default=None)
    p2.add_argument("--n", type=int, default=int(os.environ.get("TARGETS", "100")))
    p2.set_defaults(func=cmd_mmqa_index)

    p3 = sub.add_parser("webqa-manifest", help="build target_manifest.json for WebQA")
    p3.add_argument("--source", default=None, help="WebQA-targets.json")
    p3.add_argument("--test", default=None, help="WebQA_test_image.json (for Qcate)")
    p3.add_argument("--out", default=None)
    p3.set_defaults(func=cmd_webqa_manifest)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
