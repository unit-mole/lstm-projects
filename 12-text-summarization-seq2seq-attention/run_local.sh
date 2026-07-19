#!/usr/bin/env bash
set -euo pipefail
export KERAS_BACKEND=jax
python -m streamlit run app/streamlit_app.py
