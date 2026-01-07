FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates tzdata \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY neural_memory.py /app/neural_memory.py
COPY prompt.py /app/prompt.py
COPY promot.py /app/promot.py
COPY webui_app.py /app/webui_app.py
COPY webui_static /app/webui_static

ENV DB_PATH=/data/mem.db
ENV ARCHIVE_PATH=/data/archive.jsonl
ENV MAX_BYTES=5GB
ENV PSY_DEBUG=1
ENV NM_USER=ahui69
ENV NM_PROJECT=grok
ENV GROK_MODEL=grok-4-1-fast-reasoning

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "webui_app:app", "--host", "0.0.0.0", "--port", "8080"]
