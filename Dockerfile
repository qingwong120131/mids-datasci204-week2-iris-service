FROM python:3.12-slim

# Install uv inside the container from the official binary image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

# Copy dependency + metadata files first for optimal layer caching
COPY pyproject.toml uv.lock README.md ./

# Install third-party dependencies only — this layer is cached unless
# pyproject.toml/uv.lock change, even if source code changes constantly
RUN uv sync --frozen --no-dev --no-install-project

# Now copy application source and the trained model artifact
COPY src/ src/
COPY models/ models/

# Install the local project itself into the virtual environment
RUN uv sync --frozen --no-dev

# Run the venv's binaries directly at runtime, rather than via `uv run` —
# this avoids uv attempting an implicit re-sync (and a network call) on container start
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "iris_service.main:app", "--host", "0.0.0.0", "--port", "8000"]