#!/usr/bin/env bash
# ============================================================================
# run_all.sh -- run the pipeline for one benchmark, start to finish.
#
#   bash scripts/run_all.sh --bench mmqa                 # evaluate with an existing payload
#   bash scripts/run_all.sh --bench mmqa --payload       # render the images first
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_common.sh" "$@"
export BENCH TASK

while [ $# -gt 0 ]; do
  case "$1" in
    --payload) GENERATE_PAYLOAD=1; shift ;;
    --rows) T1_ROWS="$2"; shift 2 ;;
    --rows=*) T1_ROWS="${1#*=}"; shift ;;
    *) shift ;;
  esac
done
export GENERATE_PAYLOAD T1_ROWS

step() { log "== $1"; bash "$ROOT/scripts/$1" --bench "$BENCH"; }

step 00_check_environment.sh
step 01_build_retrieval_index.sh
step 02_build_target_queries.sh

if [ "${GENERATE_PAYLOAD:-0}" = "1" ]; then
  step 03_build_poisoned_images.sh
  step 03b_build_reference_pool.sh
else
  require_file "$POISON_JSON" "poison json (set GENERATE_PAYLOAD=1 to build it)"
fi

step 04_train_projection_edit.sh
step 05_main_results.sh
step 99_verify_main.sh

log "done: $RES/main_${BENCH}.tsv"
