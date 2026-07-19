@echo off
setlocal
cd /d "%~dp0"
set KERAS_BACKEND=jax
python -m streamlit run app\streamlit_app.py
endlocal
