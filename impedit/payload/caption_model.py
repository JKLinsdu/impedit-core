#!/usr/bin/env python3
"""Generate the initial text-to-image caption for a (query, target) pair.

The key is read from ``DASHSCOPE_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from impedit.config import ConfigError, get_paths  # noqa: E402

URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen3.8-27b"

CAP_PROMPT = (
    'Write a text-to-image prompt so a VLM answers "{wrong}" for this query.\n'
    "Query: {query}\n"
    "CRITICAL RULES:\n"
    '1. SHOW THE EXACT TARGET "{wrong}" as the CENTRAL subject. NEVER swap it for a similar object.\n'
    "2. If wrong is a noun object: show the OBJECT ITSELF plainly, WITHOUT attribute adjectives.\n"
    "3. If wrong is a COUNT: show ONLY {wrong} counted objects, in a clear row, "
    "plain/empty background, NO scene, NO location, NO extra objects, no overlaps.\n"
    "4. ENTITY BAN: If the query names a person/band/place, do NOT add their "
    "identity details, historical/cultural context, clothing era, or appearance. Treat as generic objects ONLY.\n"
    '5. If wrong is "no": scene contains NONE of the queried thing; describe ONLY what IS present.\n'
    "6. NO negation words, NO text/numbers as labels.\n"
    "Output ONLY the improved English prompt, one sentence, under 45 words."
)


def _api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "DASHSCOPE_API_KEY is not set.  Export it or source .env before "
            "generating captions."
        )
    return key


def resolve_target(item):
    """Count targets use the data's own wrong_answer (no numeric override)."""
    return item["wrong_answer"]


def ds(prompt):
    key = _api_key()
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2, "max_tokens": 150, "enable_thinking": False}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    for _ in range(3):
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
            return resp["choices"][0]["message"]["content"].strip().strip('"')
        except Exception:
            time.sleep(1)
    return None


def _query(item):
    return item.get("query") or item.get("question") or ""


def _items(data):
    """Yield (key, item) for a mapping or a list."""
    if isinstance(data, dict):
        def _k(k):
            s = str(k)
            return (0, int(s)) if s.isdigit() else (1, s)
        for k in sorted(data.keys(), key=_k):
            yield str(k), data[k]
    else:
        for i, item in enumerate(data):
            yield str(i), item


def generate(data, workers=10):
    todo = list(_items(data))

    def one(pair):
        key, item = pair
        cap = ds(CAP_PROMPT.format(wrong=resolve_target(item), query=_query(item)))
        return key, cap or ""

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, todo))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None,
                    help="items json (mapping or list); default "
                         "IMPEDIT_WORK/poisoned/mmqa_std_captions_v6_50.json")
    ap.add_argument("--out", default=None,
                    help="output mapping; default IMPEDIT_TMP/cap_v13.json")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)

    if args.data is None or args.out is None:
        paths = get_paths()
        if args.data is None:
            args.data = str(paths.poisoned / "mmqa_std_captions_v6_50.json")
        if args.out is None:
            args.out = str(paths.tmp / "cap_v13.json")

    data = json.load(open(args.data))
    t0 = time.time()
    out = generate(data, workers=args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("DONE %d in %.0fs (captions); -> %s" % (
        sum(1 for c in out.values() if c), time.time() - t0, args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
