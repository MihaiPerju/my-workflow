FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps, use frozen lockfile)
RUN uv sync --frozen --no-dev

# Copy source code (bust cache: 2026-09-29e)
COPY src/ ./src/
COPY worker.py ./

# Run the worker
CMD ["uv", "run", "python", "worker.py"]
