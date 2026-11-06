#!/usr/bin/env bash
# ============================================================================
# 00_check_environment.sh -- preflight: roots, imports, GPU, artifacts.
# Exits 0 only if the hard requirements hold.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_common.sh" "$@"

log "config: $CFG"
"$PY" -m impedit.config

echo
echo "--- protocol ($BENCH) ---"
printf '%-18s %s\n' \
  BENCH "$BENCH" TASK "$TASK" PROTOCOL_K "$PROTOCOL_K" CLIP_TOPK "$CLIP_TOPK" \
  RERANK_TOPK "$RERANK_TOPK" EDIT_LAMBDA "$EDIT_LAMBDA" \
  EDIT_STEPS "$EDIT_STEPS" INDEX_FILE "$INDEX_FILE" POISON_JSON "$POISON_JSON" \
  EDIT_FILE "$EDIT_FILE"

echo
echo "--- scoring ---"
"$PY" - <<'PY' || die "impedit.eval.metrics failed to import (install spacy and en_core_web_sm)"
from impedit.eval.metrics import mmqa_metrics_approx, webqa_metrics_approx
print("metrics: ok")
PY

echo
echo "--- python imports ---"
"$PY" - <<'PY'
import importlib, sys
mods = ["torch", "transformers", "faiss", "numpy", "PIL", "open_clip", "sklearn", "matplotlib", "tqdm"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("  %-14s %s" % (m, getattr(mod, "__version__", "ok")))
    except Exception as e:
        bad.append((m, str(e)[:120]))
        print("  %-14s MISSING (%s)" % (m, str(e)[:120]))
if bad:
    sys.exit("missing python packages: " + ", ".join(m for m, _ in bad))
PY

echo
echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  for g in ${GPU_POOL:-0 1 2 3}; do
    f="$(gpu_free_mib "$g")"
    [ -n "$f" ] && printf '  GPU%s free: %s MiB\n' "$g" "$f"
  done
else
  log "nvidia-smi not found; the evaluator will fall back to CPU (not supported for the paper numbers)"
fi

echo
echo "--- required inputs ---"
for pair in "INDEX_FILE:$INDEX_FILE" "POISON_JSON:$POISON_JSON" "EDIT_FILE:$EDIT_FILE"; do
  name="${pair%%:*}"; path="${pair#*:}"
  if [ -e "$path" ]; then
    log "$name ok: $path"
  else
    log "WARN $name missing: $path"
  fi
done

log "environment check complete"
