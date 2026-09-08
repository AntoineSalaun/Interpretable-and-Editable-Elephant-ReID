#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
STREAMLIT_BIN="$SCRIPT_DIR/.venv/bin/streamlit"
DATA_FILE="$SCRIPT_DIR/data/mara_embedding_100_top50.parquet"
PORT="8501"
ADDRESS="127.0.0.1"
REBUILD="0"

for arg in "$@"; do
  case "$arg" in
    --rebuild)
      REBUILD="1"
      ;;
    --port=*)
      PORT="${arg#*=}"
      ;;
    --address=*)
      ADDRESS="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--rebuild] [--port=8501] [--address=127.0.0.1]" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$VENV_PY" || ! -x "$STREAMLIT_BIN" ]]; then
  echo "Missing local viewer environment at $SCRIPT_DIR/.venv" >&2
  echo "Install it with: $SCRIPT_DIR/.venv/bin/pip install -r $SCRIPT_DIR/requirements.txt" >&2
  exit 1
fi

if [[ "$REBUILD" == "1" || ! -f "$DATA_FILE" ]]; then
  BUILD_ARGS=()
  if [[ "$REBUILD" == "1" ]]; then
    BUILD_ARGS+=(--force)
  fi
  "$VENV_PY" "$SCRIPT_DIR/build_viewer_data.py" "${BUILD_ARGS[@]}"
fi

HOSTNAME_VALUE="$(hostname 2>/dev/null || echo remote-cluster-host)"
echo "Starting Mara Embedding Atlas viewer on $HOSTNAME_VALUE"
echo "Server bind address: $ADDRESS"
echo "Server port: $PORT"
if [[ "$ADDRESS" == "127.0.0.1" || "$ADDRESS" == "localhost" ]]; then
  echo
  echo "If you are connecting from your laptop, create an SSH tunnel like:"
  echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} <username>@${HOSTNAME_VALUE}"
  echo
  echo "Then open this URL on your laptop:"
  echo "  http://127.0.0.1:${PORT}"
  echo
fi

exec "$STREAMLIT_BIN" run "$SCRIPT_DIR/app.py" \
  --server.address "$ADDRESS" \
  --server.port "$PORT"
