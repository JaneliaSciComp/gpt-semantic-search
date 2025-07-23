FROM python:3.10-slim

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

# Clone repository
RUN git clone --branch $GIT_TAG --depth 1 https://github.com/JaneliaSciComp/gpt-semantic-search.git .

# Remove unnecessary files
RUN find /app/pages -maxdepth 1 -type f \( -name '5*' -o -name '4*' \) -exec rm -f {} +

# Install dependencies
RUN pixi install

# Set environment defaults
ENV WEAVIATE_URL=http://weaviate:8080
ENV PYTHONUNBUFFERED=1

# Metadata
LABEL \
    org.opencontainers.image.title="Semantic Search Slack Bot" \
    org.opencontainers.image.description="Slack bot for semantic search" \
    org.opencontainers.image.version=${GIT_TAG} \
    org.opencontainers.image.source="https://github.com/JaneliaSciComp/gpt-semantic-search"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["pixi", "run", "python", "slack_app.py"]
