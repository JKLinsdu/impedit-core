#!/usr/bin/env bash
# ============================================================================
# 03_build_poisoned_images.sh -- regenerate the poisoned images.
# Needs the renderer (~20 GB) and one surrogate worker (~19 GB) running; the
# commands to start them are at the end of this header.  Several hours per
# surrogate.  The released payload makes this step optional.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_common.sh" "$@"

RENDER_CACHE="${RENDER_CACHE:-$WORK/render_cache}"
IDXS="${IDXS:-$(seq -s, 0 99)}"
SURROGATES="${SURROGATES:-qwen llava qwen2vl internvl3 idefics3}"

"$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('impedit.payload.generate_verify_rewrite') else 1)" \
  || die "impedit.payload.generate_verify_rewrite is not importable"
"$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('impedit.payload.renderer') else 1)" \
  || die "impedit.payload.renderer is not importable"

if [ -n "${SURROGATE_LOG_DIR:-}" ]; then
  for sg in $SURROGATES; do
    surrogate_log_ready "$sg" || die "VLM worker for $sg is not loaded (log: $SURROGATE_LOG_DIR/vlm_$sg.log)"
  done
fi

mkdir -p "$RENDER_CACHE" "$LOGS" "$SURROGATE_POISON"

if [ "${START_RENDER_SERVER:-1}" = "1" ]; then
  log "starting impedit.payload.renderer (log: $LOGS/render_server.out)"
  ( cd "$ROOT" && RENDER_OUT="$RENDER_CACHE" RENDER_GPU="${RENDER_GPU:-0}" \
      setsid nohup "$PY" -u -m impedit.payload.renderer \
      > "$LOGS/render_server.out" 2>&1 < /dev/null & echo $! > "$TMP/render_server.pid" )
  for _ in $(seq 1 90); do
    grep -q '^loaded' "$LOGS/render_server.out" 2>/dev/null && break
    sleep 20
  done
  grep -q '^loaded' "$LOGS/render_server.out" 2>/dev/null || die "render_server did not report 'loaded'"
fi

if [ "$TASK" = "MMQA" ]; then
  [ -n "${DASHSCOPE_API_KEY:-}" ] || log "  WARN DASHSCOPE_API_KEY unset: rendering without caption rewrites"
  # Optional: a pool of seed images (q<i>.png) used for the first attempt only.
  # Without it the loop renders every attempt from the caption.
  if [ -n "${PAYLOAD_BASE_IMAGES:-}" ]; then
    require_dir "$PAYLOAD_BASE_IMAGES" "seed image pool (PAYLOAD_BASE_IMAGES)"
  fi
  for sg in $SURROGATES; do
    OUTDIR="$PAYLOAD/mmqa_surrogate_${sg}_100"
    OUTJSON="$SURROGATE_POISON/poison_${sg}.json"
    if [ "$(ls "$OUTDIR" 2>/dev/null | wc -l)" -ge 100 ] && [ -s "$OUTJSON" ]; then
      log "  $sg already complete, skipping"
      continue
    fi
    log "  generating MMQA payload: $sg -> $OUTDIR"
    mkdir -p "$OUTDIR"
    ( cd "$ROOT" && SURROGATE="$sg" SURROGATE_SERVER="$sg" PAYLOAD_BASE_IMAGES="${PAYLOAD_BASE_IMAGES:-}" \
        PAYLOAD_OUT_JSON="$OUTJSON" DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
        "$PY" -u -m impedit.payload.generate_verify_rewrite mmqa "$IDXS" "$OUTDIR" ) \
      > "$LOGS/payload_${sg}.log" 2>&1
    [ "$sg" = "qwen" ] && cp -f "$OUTJSON" "$POISON_JSON"
  done
else
  WQ_BASE="${WQ_BASE:-$IMPEDIT_DATA/webqa/WebQA-targets.json}"
  require_file "$WQ_BASE" "WebQA target source (set WQ_BASE)"
  for sg in $SURROGATES; do
    WQ_SRC="$SURROGATE_POISON/WebQA_${sg}.json"
    cp -f "$WQ_BASE" "$WQ_SRC"
    OUTDIR="$PAYLOAD/webqa_surrogate_${sg}_100"
    OUTJSON="$SURROGATE_POISON/poison_${sg}.json"
    if [ -s "$OUTJSON" ] && [ "$("$PY" -c "import json;print(len(json.load(open('$OUTJSON'))))" 2>/dev/null || echo 0)" -ge 100 ]; then
      log "  $sg already complete, skipping"
      continue
    fi
    log "  generating WebQA payload: $sg -> $OUTDIR"
    mkdir -p "$OUTDIR"
    ( cd "$ROOT" && SURROGATE="$sg" SURROGATE_SERVER="$sg" WEBQA_SGIR="$WQ_SRC" \
        DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}" \
        "$PY" -u -m impedit.payload.generate_verify_rewrite webqa "$IDXS" "$OUTDIR" ) \
      > "$LOGS/payload_${sg}.log" 2>&1
    "$PY" -u -m impedit.data.build_poison_manifest webqa --source "$WQ_SRC" --out "$OUTJSON" \
      >> "$LOGS/payload_${sg}.log" 2>&1
    [ "$sg" = "qwen" ] && cp -f "$OUTJSON" "$POISON_JSON"
  done
fi

if [ "${START_RENDER_SERVER:-1}" = "1" ] && [ -f "$TMP/render_server.pid" ]; then
  kill "$(cat "$TMP/render_server.pid")" 2>/dev/null || true
  rm -f "$TMP/render_server.pid"
fi

log "payload generation complete"
