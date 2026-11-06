#!/usr/bin/env bash
# ============================================================================
# 03b_build_reference_pool.sh -- render one stand-in image per target query.
# Writes $IMPEDIT_WORK/ref_pool_gt_<bench>.json, read by 04.  The renderer must
# be running.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/_common.sh" "$@"

"$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('impedit.data.build_reference_pool') else 1)" \
  || die "impedit.data.build_reference_pool is not importable"

"$PY" -u -m impedit.data.build_reference_pool --task "$BENCH"
