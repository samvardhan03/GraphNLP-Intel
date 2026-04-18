FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps for spaCy and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[serve]"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY graphnlp ./graphnlp
COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "graphnlp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
