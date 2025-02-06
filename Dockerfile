# Use Python 3.12 base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

ENV POETRY_CACHE_DIR=/tmp/poetry_cache

# Install poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install dependencies using poetry
RUN --mount=type=cache,target=$POETRY_CACHE_DIR  poetry config virtualenvs.create false \
    && poetry install --no-root

COPY grpo_server ./grpo_server
COPY test_data ./test_data

# Expose the port the app runs on
EXPOSE 8000

# Environment variables
ENV API_KEY=default_key
ENV OUTPUT_DIR=/app/output

# Create output directory
RUN mkdir -p /app/output
# Command to run with uvicorn_main
# CMD ["python", "-m", "grpo_server.uvicorn_main", "--api-key", "$API_KEY", "--output-dir", "$OUTPUT_DIR"]
CMD ["python", "-m", "grpo_server.uvicorn_main" ]
