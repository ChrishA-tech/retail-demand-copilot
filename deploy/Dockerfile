FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY artifacts ./artifacts
ENV PORT=8080
EXPOSE 8080
# Vertex AI passes AIP_HTTP_PORT / AIP_HEALTH_ROUTE / AIP_PREDICT_ROUTE; defaults work locally
CMD ["sh", "-c", "uvicorn rdc.api:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-$PORT}"]
