#!/usr/bin/env bash
# Start mid-voice on http://127.0.0.1:7860
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "No .venv found — run:  uv venv --python 3.12 && uv pip install -e '.[tts,mt]'" >&2
  exit 1
fi

exec .venv/bin/python -m app.main "$@"
