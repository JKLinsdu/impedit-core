#!/usr/bin/env bash
# ============================================================================
# setup.sh -- venv, dependencies, and .env.
#
#   bash setup.sh --data /data/impedit/data --models /data/impedit/models
#
# Options: --data DIR  --models DIR  --work DIR  --hf-home DIR
#          TORCH_INDEX=... for another CUDA build
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
DATA="" MODELS="" WORK="" HFH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --data)    DATA="$2";   shift 2 ;;
    --models)  MODELS="$2"; shift 2 ;;
    --work)    WORK="$2";   shift 2 ;;
    --hf-home) HFH="$2";    shift 2 ;;
    *) shift ;;
  esac
done
for d in "$DATA" "$MODELS"; do
  [ -n "$d" ] && [ ! -d "$d" ] && { echo "no such directory: $d" >&2; exit 1; }
done

[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
[ -f .env ] || cp .env.example .env
python - "$DATA" "$MODELS" "$WORK" "$HFH" "$ROOT" <<'PY'
import sys
from pathlib import Path

data, models, work, hfh, root = sys.argv[1:6]
env = Path(root) / ".env"
updates = {
    "IMPEDIT_DATA": data,
    "IMPEDIT_MODELS": models,
    "IMPEDIT_WORK": work or str(Path(root) / "work"),
    "IMPEDIT_HF_HOME": hfh or str(Path.home() / ".cache" / "huggingface"),
    "PYTHON": str(Path(root) / ".venv" / "bin" / "python"),
}
updates = {k: v for k, v in updates.items() if v}
out, seen = [], set()
for line in env.read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k = line.split("=", 1)[0].strip()
        if k in updates:
            out.append("%s=%s" % (k, updates[k]))
            seen.add(k)
            continue
    out.append(line)
out += ["%s=%s" % (k, v) for k, v in updates.items() if k not in seen]
env.write_text("\n".join(out) + "\n")
print("wrote", env)
PY

python -m pip install -q --upgrade pip
python -m pip install torch --index-url "$TORCH_INDEX"
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm

if [ -n "$DATA" ] && [ -n "$MODELS" ]; then
  python -m impedit.config
else
  echo
  echo "fill in IMPEDIT_DATA and IMPEDIT_MODELS in .env, then:"
  echo "  bash scripts/run_all.sh --bench mmqa"
fi
