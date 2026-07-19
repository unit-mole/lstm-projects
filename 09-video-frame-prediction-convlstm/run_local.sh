#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export KERAS_BACKEND="${KERAS_BACKEND:-jax}"
python -m streamlit run app/streamlit_app.py
