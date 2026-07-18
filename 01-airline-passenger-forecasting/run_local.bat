@echo off
setlocal
cd /d "%~dp0"
set KERAS_BACKEND=jax

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.12 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app\streamlit_app.py
