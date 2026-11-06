#!/usr/bin/env bash
# ============================================================================
# 01_build_retrieval_index.sh -- verify the CLIP FAISS index, rebuild if missing.
# REBUILD_INDEX=1 forces a rebuild (~5-15 min, ~3 GB VRAM).
# Writes $INDEX_FILE and $IMPEDIT_DATA/<bench>/<TASK>_test_image_index_to_id.json.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_common.sh" "$@"

IDJSON="$("$PY" - "$TASK" <<'PY'
import sys
from impedit.config import get_paths
from impedit.eval.paths import dataset_files
print(dataset_files(get_paths(), sys.argv[1])["index_to_id"])
PY
)"

REPORT="$LOGS/index_report.txt"
: > "$REPORT"

verify_index() {
  "$PY" - "$INDEX_FILE" "$IDJSON" "$TASK" <<'PY' | tee -a "$REPORT"
import sys
from pathlib import Path
import faiss, json
index_path, id_path, task = sys.argv[1], sys.argv[2], sys.argv[3]
for label, p in (("index", index_path), ("id json", id_path)):
    if not Path(p).exists():
        sys.exit("MISSING %s: %s" % (label, p))
idx = faiss.read_index(index_path)
ids = json.load(open(id_path))
print("%s index: ntotal=%d dim=%d" % (task, idx.ntotal, idx.d))
print("%s id json: entries=%d" % (task, len(ids)))
if idx.ntotal != len(ids):
    print("WARN: index.ntotal != id-json entries; index.add() appends poisons at run time,")
    print("      so equality is expected only for a pristine index.")
print("OK")
PY
}

if [ "${REBUILD_INDEX:-0}" = "1" ] || [ ! -f "$INDEX_FILE" ]; then
  log "building the CLIP index with impedit.data.build_retrieval_index"
  "$PY" -u -m impedit.data.build_retrieval_index --task "$BENCH" --retriever clip | tee -a "$REPORT"
fi

if [ ! -f "$INDEX_FILE" ]; then
  log "FATAL: no index at $INDEX_FILE"
  log "# TODO(repro): build the index first (see above) or point INDEX_FILE at one."
  exit 1
fi

require_file "$IDJSON" "index id mapping"
verify_index

if [ "${REBUILD_OTHER:-0}" = "1" ]; then
  TASK_LOWER="$(printf '%s' "$TASK" | tr '[:upper:]' '[:lower:]')"
  for r in openclip siglip; do
    log "building the $r index"
    "$PY" -u -m impedit.data.build_retrieval_index --task "$TASK_LOWER" --retriever "$r" \
      >> "$REPORT" 2>&1 || log "# TODO(repro): $r index build failed; see $REPORT"
  done
fi

log "index check complete; report: $REPORT"
