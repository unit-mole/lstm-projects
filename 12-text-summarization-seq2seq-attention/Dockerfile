FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=jax \
    PORT=8501

WORKDIR /app
COPY app/requirements.txt /tmp/app-requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/app-requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["sh", "-c", "python -m streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT}"]
