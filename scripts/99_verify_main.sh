#!/usr/bin/env bash
# ============================================================================
# 99_verify_main.sh -- print the norerank ASR/ACC/RSR and check they parse.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_common.sh" "$@"

"$PY" - "$RES/main_${BENCH}.tsv" <<'PY'
import sys

tsv = sys.argv[1]
row = None
for line in open(tsv, encoding="utf-8"):
    c = line.rstrip("\n").split("\t")
    if c and c[0] == "norerank":
        row = {"ASR": float(c[1]), "ACC": float(c[2]), "RSR": float(c[3])}
if row is None:
    raise SystemExit("missing row 'norerank' in %s" % tsv)
bad = [k for k, v in row.items() if not 0.0 <= v <= 1.0]
if bad:
    raise SystemExit("out of range: %s" % ", ".join(bad))
for k in ("ASR", "ACC", "RSR"):
    print("%-4s %.4f" % (k, row[k]))
PY
