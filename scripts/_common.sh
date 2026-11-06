# shellcheck shell=bash
# shellcheck shell=bash
# ============================================================================
# Shared helpers for every entry point.  Sources ./.env (optional) then
# ./configs/${BENCH}.env (required, plain shell, sourced with set -a).
# ============================================================================

ROOT="${IMPEDIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

if [ -f "$ROOT/.env" ]; then
  set -a
  . "$ROOT/.env"
  set +a
fi

# -b/--bench <name>; the environment variable still works as a fallback.
# Reads a copy of the arguments so the caller keeps its own "$@".
_args=("$@")
_i=0
while [ "$_i" -lt "${#_args[@]}" ]; do
  case "${_args[$_i]}" in
    -b|--bench) BENCH="${_args[$((_i + 1))]:-}"; _i=$((_i + 2)) ;;
    --bench=*)  BENCH="${_args[$_i]#*=}"; _i=$((_i + 1)) ;;
    *) _i=$((_i + 1)) ;;
  esac
done

BENCH="${BENCH:-mmqa}"
CFG="${IMPEDIT_CONFIG:-$ROOT/configs/${BENCH}.env}"
if [ ! -f "$CFG" ]; then
  echo "FATAL: missing $CFG" >&2
  echo "       configs/${BENCH}.env ships with the repository; set IMPEDIT_CONFIG to point elsewhere" >&2
  exit 1
fi
set -a
. "$CFG"
set +a

PY="${PYTHON:-python3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "$BENCH" in
  mmqa|MMQA) BENCH=mmqa; TASK=MMQA ;;
  webqa|WebQA) BENCH=webqa; TASK=WebQA ;;
  *) echo "FATAL: BENCH must be mmqa or webqa, got '$BENCH'" >&2; exit 1 ;;
esac

_cfg_path() {
  "$PY" - "$1" <<'PY'
import sys
from impedit.config import get_paths
print(getattr(get_paths(), sys.argv[1]))
PY
}
_cfg_index() {
  "$PY" - "$1" <<'PY'
import sys
from pathlib import Path
from impedit.config import get_paths
print(Path(get_paths().indexes) / ("%s_test_image_clip.index" % sys.argv[1]))
PY
}

WORK="$(_cfg_path work)"
RES="$WORK/results/$BENCH"
LOGS="$WORK/logs/$BENCH"
EDITS="$WORK/edits"
SURROGATE_EDITS="$EDITS/surrogates/$BENCH"
PAYLOAD="$WORK/poisoned"
SURROGATE_POISON="$PAYLOAD/surrogates/$BENCH"
TMP="$WORK/tmp"
mkdir -p "$RES" "$LOGS" "$EDITS" "$SURROGATE_EDITS" "$PAYLOAD" "$SURROGATE_POISON" "$TMP"

if [ "$TASK" = "MMQA" ]; then
  PROTOCOL_K="${PROTOCOL_K:-1}"
  RERANK_TOPK="${RERANK_TOPK:-5}"
  POISON_JSON="${POISON_JSON:-$PAYLOAD/poison_mmqa.json}"
  EDIT_FILE="${EDIT_FILE:-$EDITS/mm_prod_800.pt}"
else
  PROTOCOL_K="${PROTOCOL_K:-2}"
  RERANK_TOPK="${RERANK_TOPK:-5}"
  POISON_JSON="${POISON_JSON:-$PAYLOAD/poison_webqa.json}"
  EDIT_FILE="${EDIT_FILE:-$EDITS/wq_prod_800.pt}"
fi
CLIP_TOPK="${CLIP_TOPK:-$PROTOCOL_K}"
EDIT_LAMBDA="${EDIT_LAMBDA:-0.01}"
EDIT_STEPS="${EDIT_STEPS:-800}"
SURROGATE="${SURROGATE:-qwen}"
INDEX_FILE="${INDEX_FILE:-$(_cfg_index "$TASK")}"
SURROGATES="${SURROGATES:-qwen}"
TRAINER_MODULE="${TRAINER_MODULE:-impedit.edit.train_projection_edit}"
TRAINER="${TRAINER:-}"   # optional explicit file path; empty = use TRAINER_MODULE

log()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die()  { printf 'FATAL: %s\n' "$*" >&2; exit 1; }
require_file() { [ -f "$1" ] || die "missing $2: $1"; }
require_dir()  { [ -d "$1" ] || die "missing $2: $1"; }

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null \
    | tr -d ' ' | awk -F, '{print $1-$2}'
}

pick_gpu() {
  local need="$1"; shift
  local pool="${*:-${GPU_POOL:-0 1 2 3}}"
  local g free best="" bestfree=0
  for g in $pool; do
    free="$(gpu_free_mib "$g")"
    [ -z "$free" ] && continue
    if [ "$free" -gt "$bestfree" ]; then bestfree="$free"; best="$g"; fi
  done
  if [ -n "$best" ] && [ "$bestfree" -ge "$need" ]; then
    echo "$best"
  else
    echo ""
  fi
}

run_evaluator() {
  local logf="$1"; shift
  "$PY" -u -m impedit.eval.evaluate "$@" > "$logf" 2>&1
}

run_score() {
  local logf="$1"; shift
  "$PY" -u -m impedit.eval.compute_metrics "$@" > "$logf" 2>&1
}

metric() {
  grep -aE "$2" "$1" 2>/dev/null | head -1 | grep -oE '[-+]?[0-9]+\.[0-9]+' | head -1
}

metric_tail() {
  grep -aE "$2" "$1" 2>/dev/null | tail -1 | grep -oE '[-+]?[0-9]+\.[0-9]+' | tail -1
}

rsr_from_log() {
  local v
  v="$(grep -a 'Retrieval poisoned recall:' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)"
  [ -z "$v" ] && { echo "NA"; return; }
  awk -v v="$v" -v d="$2" 'BEGIN{printf "%.4f", v/d}'
}

rsr_from_save() {
  local d="$WORK/results/$1" f v
  [ -d "$d" ] || { echo "NA"; return; }
  f="$(find "$d" -name 'k*_usecap*.txt' 2>/dev/null | head -1)"
  [ -z "$f" ] && { echo "NA"; return; }
  v="$(grep -a 'Retrieval poisoned recall:' "$f" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)"
  [ -z "$v" ] && { echo "NA"; return; }
  awk -v v="$v" -v d="$2" 'BEGIN{printf "%.4f", v/d}'
}

surrogate_log_ready() {
  local m="$1"
  if [ -n "${SURROGATE_LOG_DIR:-}" ]; then
    grep -q '^loaded' "$SURROGATE_LOG_DIR/vlm_$m.log" 2>/dev/null
  else
    return 0
  fi
}

trainer_ready() {
  "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$TRAINER_MODULE') else 1)"
}

train_edit() {
  local task="$1" pj="$2" seed="$3" steps="$4" out="$5" logf="$6"; shift 6
  local CMD
  if [ -n "$TRAINER" ]; then
    [ -f "$TRAINER" ] || { echo "# TODO(repro): trainer not found at $TRAINER" >&2; return 1; }
    CMD=("$PY" -u "$TRAINER")
  else
    trainer_ready || {
      echo "# TODO(repro): trainer module $TRAINER_MODULE not importable" >&2
      echo "#              expected impedit/edit/train.py (CLI: --task/--poison_json/--seed/--steps/--out)" >&2
      return 1
    }
    CMD=("$PY" -u -m "$TRAINER_MODULE")
  fi
  env "$@" EDIT_GPU="${EDIT_GPU:-${EVAL_GPU:-0}}" CUDA_VISIBLE_DEVICES="${EDIT_GPU:-${EVAL_GPU:-0}}" \
    "${CMD[@]}" --task "$task" --poison_json "$pj" --seed "$seed" --steps "$steps" --out "$out" \
    > "$logf" 2>&1
}

summarize_score() {
  local f="$1" label="$2" rsr="${3:-NA}"
  local asr acc
  asr="$(metric "$f" 'attack-ASR')"
  acc="$(metric "$f" 'attack-ACC')"
  printf '%s\t%s\t%s\t%s\n' "$label" "${asr:-NA}" "${acc:-NA}" "$rsr"
}
