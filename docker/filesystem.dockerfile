FROM python:3.9-slim

ARG GIT_TAG=main

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pixi package manager
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

WORKDIR /app

# Copy dependency files first for better caching
COPY pixi.toml pixi.lock ./

# Add platform and install dependencies
RUN pixi project platform add linux-aarch64 || true && \
    pixi install

# Copy application files
COPY filesystem_rag_service.py ./
COPY indexing/ indexing/
COPY scraping/ scraping/

# Create required directories
RUN mkdir -p /app/logs /workspace

# Set up environment defaults
ENV WEAVIATE_URL=http://weaviate:8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Metadata
LABEL \
    org.opencontainers.image.title="Filesystem RAG CLI" \
    org.opencontainers.image.description="CLI service for filesystem-based RAG operations" \
    org.opencontainers.image.version=${GIT_TAG} \
    org.opencontainers.image.source="https://github.com/JaneliaSciComp/gpt-semantic-search"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default entrypoint and command
ENTRYPOINT ["pixi", "run", "python", "filesystem_rag_service.py"]
CMD ["--help"]