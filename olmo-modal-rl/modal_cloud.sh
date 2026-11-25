#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-/workspace/.local/share/uv/tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-/workspace/.local/bin}"
# Install once with:
# uv tool install --python 3.13 --with truststore==0.10.5 'modal[api-proxy-support]==1.6.4'
exec uv run --no-project --python "$UV_TOOL_DIR/modal/bin/python" python -c \
  'import truststore; truststore.inject_into_ssl(); import runpy; runpy.run_module("modal", run_name="__main__")' "$@"
