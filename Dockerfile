FROM python:3.9-slim

ARG GIT_TAG=main
ENV WEAVIATE_URL=http://localhost:8080

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --branch $GIT_TAG --depth 1 https://github.com/JaneliaSciComp/gpt-semantic-search.git .

RUN pip3 install -r requirements.txt


LABEL \
    org.opencontainers.image.title="JaneliaGPT Semantic Search" \
    org.opencontainers.image.description="Semantic search for Janelia resources" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org" \
    org.opencontainers.image.licenses="BSD-3-Clause" \
    org.opencontainers.image.version=${GIT_TAG} \
    org.opencontainers.image.source="https://github.com/JaneliaSciComp/gpt-semantic-search"

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT streamlit run 1_🔍_Search.py --server.port=8501 --server.address=0.0.0.0 -- -w $WEAVIATE_URL

