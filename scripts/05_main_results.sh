#!/usr/bin/env bash
# ============================================================================
# 05_main_results.sh -- evaluate the five IMPEdit rows of Table I.
# Writes $RES/main_<bench>_<row>.txt and $RES/main_<bench>.tsv.
# ~1-10 min per row; ~30-60 min per benchmark.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_common.sh" "$@"

require_file "$INDEX_FILE" "FAISS index"
require_file "$POISON_JSON" "poison json"
require_file "$EDIT_FILE" "trained edit"

SUMMARY="$RES/main_${BENCH}.tsv"
: > "$SUMMARY"
printf 'row\tASR\tACC\tRSR\n' >> "$SUMMARY"

score_row() {  # <tag> <evaluator log> <save_dir>
  local tag="$1" logf="$2" save="$3"
  local sf="$RES/main_${BENCH}_${tag}.txt"
  run_score "$sf" --task "$TASK" --poison_json "$POISON_JSON" \
    --save_dir "$save" --gen "${GEN:-qwen}"
  local rsr
  rsr="$(rsr_from_save "$save" "${NUM_P:-$PROTOCOL_K}")"
  summarize_score "$sf" "$tag" "$rsr" | tee -a "$SUMMARY"
}

run_norerank() {
  local tag="norerank" save="results_T1_${BENCH}_norerank"
  local logf="$LOGS/main_${BENCH}_${tag}.log"
  local gpu="${EVAL_GPU:-0}"
  log "Table I [$BENCH/$tag] GPU$gpu"
  rm -rf "${WORK:?}/results/$save"
  env PROJECTION_EDIT="$EDIT_FILE" CUDA_VISIBLE_DEVICES="$gpu" EVAL_GPU="$gpu" MLLM_SINGLE_GB="${MLLM_SINGLE_GB:-20}" \
    "$PY" -u -m impedit.eval.evaluate --task "$TASK" --generator_type qwen \
    --retriever_type clip --rerank_off --clip_topk "$CLIP_TOPK" --num_p "${NUM_P:-$PROTOCOL_K}" \
    --index_file_path "$INDEX_FILE" --poisoned_data_path "$POISON_JSON" \
    --save_dir "$save" --surrogate_server qwen > "$logf" 2>&1
  GEN=qwen score_row "$tag" "$logf" "$save"
}

run_rerank() {  # <tag> <model> <use_caption 0|1>
  local tag="$1" mdl="$2" uc="$3" save="results_T1_${BENCH}_$1"
  local logf="$LOGS/main_${BENCH}_${tag}.log"
  local need=17500; [ "$mdl" = "qwen" ] && need=21500
  local gpu="${RERANK_GPU:-}"
  [ -z "$gpu" ] && gpu="$(pick_gpu "$need" ${RERANK_GPU_POOL:-0 1 2 3})"
  [ -n "$gpu" ] || die "no GPU with ${need} MiB free for $tag (set RERANK_GPU)"
  local cap=""; [ "$uc" = "1" ] && cap="--use_caption"
  log "Table I [$BENCH/$tag] $mdl/$mdl caption=$uc GPU$gpu"
  rm -rf "${WORK:?}/results/$save"
  env PROJECTION_EDIT="$EDIT_FILE" CUDA_VISIBLE_DEVICES="$gpu" EVAL_GPU="$gpu" \
    MLLM_SINGLE_GB="${MLLM_SINGLE_GB:-20}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u -m impedit.eval.evaluate --task "$TASK" --generator_type "$mdl" --reranker_type "$mdl" \
    --retriever_type clip --clip_topk "$RERANK_TOPK" --num_p "${NUM_P:-$PROTOCOL_K}" $cap \
    --index_file_path "$INDEX_FILE" --poisoned_data_path "$POISON_JSON" \
    --save_dir "$save" > "$logf" 2>&1
  grep -qa "OutOfMemoryError" "$logf" && die "$tag OOM; raise MLLM_SINGLE_GB or free the GPU"
  GEN="$mdl" score_row "$tag" "$logf" "$save"
}

for _r in ${T1_ROWS:-norerank qwen_cap0 qwen_cap1 llava_cap0 llava_cap1}; do
  case "$_r" in
    norerank)   run_norerank ;;
    qwen_cap0)  run_rerank qwen_cap0  qwen  0 ;;
    qwen_cap1)  run_rerank qwen_cap1  qwen  1 ;;
    llava_cap0) run_rerank llava_cap0 llava 0 ;;
    llava_cap1) run_rerank llava_cap1 llava 1 ;;
    *) echo "FATAL: unknown T1_ROWS entry '$_r'" >&2; exit 1 ;;
  esac
done

log "Table I complete; summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
