#!/usr/bin/env bash
# ============================================================================
# 02_build_target_queries.sh -- build the target manifests (CPU only, seconds).
#   MMQA -> $IMPEDIT_DATA/mmqa/MMQA_ImageQ_index.json
#   WebQA -> $IMPEDIT_DATA/targets/target_manifest.json
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_common.sh" "$@"

TARGETS_OUT="${TARGETS_OUT:-$IMPEDIT_DATA/targets}"
mkdir -p "$TARGETS_OUT" "$SURROGATE_POISON"

if [ "$TASK" = "MMQA" ]; then
  SRC="$IMPEDIT_DATA/mmqa/MMQA-targets-100.json"
  require_file "$SRC" "100 target designs (IMPEDIT_DATA/mmqa/MMQA-targets-100.json)"
  "$PY" -u -m impedit.data.build_target_queries mmqa-index \
    --targets "$SRC" --out "$IMPEDIT_DATA/mmqa/MMQA_ImageQ_index.json"
  log "per-surrogate poison manifests are written by 03_build_poisoned_images.sh"
else
  SRC="$IMPEDIT_DATA/webqa/WebQA-targets.json"
  require_file "$SRC" "reconciled WebQA target design (IMPEDIT_DATA/webqa/WebQA-targets.json)"
  "$PY" -u -m impedit.data.build_target_queries webqa-manifest \
    --source "$SRC" --out "$IMPEDIT_DATA/targets/target_manifest.json"
  log "WebQA poison json ($POISON_JSON) is written by 03_build_poisoned_images.sh"
fi

log "target manifests complete"
