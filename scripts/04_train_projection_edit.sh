#!/usr/bin/env bash
# ============================================================================
# 04_train_projection_edit.sh -- train the projection edit W' = R W0, 800 steps.
# Writes $EDIT_FILE and checks the target:N/N gate.  ~1-2 min, ~10-20 GB VRAM.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_common.sh" "$@"

TASK_LOWER="$(printf '%s' "$TASK" | tr '[:upper:]' '[:lower:]')"
SEEDS="${SEEDS:-0}"
GPU="${EDIT_GPU:-${EVAL_GPU:-0}}"
SUMMARY="$LOGS/train_summary.tsv"
: > "$SUMMARY"
printf 'surrogate\tseed\tsteps\tresult\tgate\tout\n' >> "$SUMMARY"

if [ -n "$TRAINER" ]; then
  [ -f "$TRAINER" ] || die "TRAINER=$TRAINER does not exist"
elif ! trainer_ready; then
  log "# TODO(repro): trainer module $TRAINER_MODULE not importable"
  log "#              expected impedit/edit/train.py (CLI: --task/--poison_json/--seed/--steps/--out)"
  exit 1
fi

for sg in $SURROGATES; do
  PJ="$SURROGATE_POISON/poison_${sg}.json"
  if [ ! -f "$PJ" ] && [ "$sg" = "qwen" ]; then PJ="$POISON_JSON"; fi
  if [ ! -f "$PJ" ]; then
    log "  $sg: missing poison json $PJ; skipping (run 02/03 first)"
    continue
  fi

  for s in $SEEDS; do
    if [ "$TASK" = "MMQA" ]; then
      E="$SURROGATE_EDITS/edit_${sg}_${s}.pt"
    else
      E="$SURROGATE_EDITS/edit_wq_${sg}_${s}.pt"
    fi
    [ "$sg" = "qwen" ] && [ "$s" = "0" ] && E="$EDIT_FILE"
    L="$LOGS/train_${sg}_s${s}.txt"

    STEPS="$EDIT_STEPS"
    log "  train $sg seed=$s steps=$STEPS -> $E"
    ENVS=(
      "ABL_WT=1" "ABL_WROT=$EDIT_LAMBDA" "ABL_WS=1" "ABL_WBR=0"
      "GAMMA_B=0.05" "PZ_TOL=1e-2" "BEN_TOPK=$PROTOCOL_K" "NT_ANCHOR=100000"
      "KB_FREE=1" "REF_POOL=${REF_POOL:-$WORK/ref_pool_gt_${BENCH}.json}" "NO_KB=1"
      "EDIT_GPU=$GPU" "CUDA_VISIBLE_DEVICES=$GPU"
    )
    [ "$TASK" = "WebQA" ] && ENVS+=("NT_ANCHOR=100000")
    if train_edit "$TASK_LOWER" "$PJ" "$s" "$STEPS" "$E" "$L" "${ENVS[@]}"; then
      BEN_TOPK=$PROTOCOL_K NT_ANCHOR=100000 CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m impedit.eval.score_edit_vs_kb \
          --task "$TASK_LOWER" --edit "$E" --poison_json "$PJ" > "$L.score" 2>&1
      T="$(sed -n 's/.*target:\([0-9]*\)\/.*/\1/p' "$L.score" | tail -1)"
      R="$(grep -a RESULT "$L.score" | tail -1 || true)"
      log "    $R"
      if [ "${T:-0}" -ge 100 ]; then
        printf '%s\t%s\t%s\t%s\tPASS\t%s\n' "$sg" "$s" "$STEPS" "$R" "$E" >> "$SUMMARY"
      else
        printf '%s\t%s\t%s\t%s\tFAIL\t%s\n' "$sg" "$s" "$STEPS" "no 100/100" "$E" >> "$SUMMARY"
        log "  WARN $sg seed=$s did not reach 100/100"
      fi
    else
      log "    training failed (see $L)"
      printf '%s\t%s\t%s\t%s\tFAIL\t%s\n' "$sg" "$s" "$STEPS" "training failed" "$E" >> "$SUMMARY"
    fi
  done
done

log "training complete; summary: $SUMMARY"
