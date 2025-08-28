# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACT_DIR=/app/artifacts

# System deps (uv installer + build tools if needed; libgomp1 for xgboost/lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# --- COPY from the build context (project root), not from docker/ ---
# Lockfiles first to leverage layer cache
COPY pyproject.toml uv.lock ./

# Pre-resolve deps (optional but speeds up later runs)
RUN uv sync --frozen

# Copy the rest of the project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Copy entrypoint (path from context root)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Healthcheck (works only when serving)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]

#docker build -t churn-mini-prod -f docker/Dockerfile .

#docker run -p 8000:8000 churn-mini-prod train --data data/customer_churn_synth.csv --outdir artifacts/
#docker run -p 8000:8000 churn-mini-prod drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv
#docker run -p 8000:8000 churn-mini-prod serve --host