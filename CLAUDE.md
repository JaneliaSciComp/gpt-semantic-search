# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a semantic search service for Janelia resources using OpenAI GPT models and Weaviate vector database. The system indexes multiple data sources (Janelia website, wiki, Slack) and provides a ChatGPT-style search interface via Streamlit.

## Common Commands

### Development Setup
```bash
# Install pixi dependencies
pixi install
pixi shell

# Start Weaviate database only
docker compose up weaviate -d

# Run the main search webapp
pixi run streamlit run 1_🔍_Search.py -- -w http://localhost:8777

# Run the no-Slack version
pixi run streamlit run No_Slack_Search.py -- -w http://localhost:8777
```

### Data Indexing
```bash
# Download wiki data
pixi run python download_wiki.py

# Index Slack export
pixi run ./index_slack.py -i ./data/slack/slack_export_Janelia-Software_ALL -c Janelia

# Index wiki data
pixi run ./index_wiki.py -i ./data/wiki -c Janelia

# Index web data
pixi run ./index_web.py -i ./data/janelia.org -c Janelia

# Run web scraper
pixi run scrapy runspider spider.py
```

### Container Management
```bash
# Full deployment (production images)
docker compose up -d

# Local development with locally built images
docker compose -f docker-compose-dev.yaml up -d

# Rebuild containers manually
docker build --no-cache . -t ghcr.io/janeliascicomp/gpt-semantic-search-web:latest
docker build --no-cache . -f Dockerfile_slack -t ghcr.io/janeliascicomp/gpt-semantic-search-slack-bot:latest
```

### Slack Scraping (Incremental)
```bash
# Flexible interval scraping - supports any time interval
pixi run python slack_scrape/slack_incremental_scraper.py --interval 3h    # Every 3 hours
pixi run python slack_scrape/slack_incremental_scraper.py --interval 30m   # Every 30 minutes  
pixi run python slack_scrape/slack_incremental_scraper.py --interval 1d    # Daily (default)

# Smart indexing - automatically finds what needs indexing
pixi run python slack_scrape/slack_incremental_indexer.py               # Auto-discover new folders
pixi run python slack_scrape/slack_incremental_indexer.py --retry-only  # Only retry failed folders

# Advanced scraping options
pixi run python slack_scrape/slack_incremental_scraper.py --interval 3h --buffer 30m  # 3h interval with 30m buffer
pixi run python slack_scrape/slack_incremental_scraper.py --start-from 1719389200     # Start from specific timestamp
pixi run python slack_scrape/slack_incremental_scraper.py --debug                     # Debug mode

# Advanced indexing options
pixi run python slack_scrape/slack_incremental_indexer.py --buffer 0.5     # 30min buffer for indexing
pixi run python slack_scrape/slack_incremental_indexer.py -i /path/to/folder  # Process specific folder

# Historical scraping (inefficient, use only if needed)
pixi run python slack_past_scraper.py
```

## Architecture Overview

### Core Components

**Main Search Interface (`1_🔍_Search.py`)**
- Primary Streamlit app with ChatGPT-style interface
- Handles user queries, manages session state, and displays results
- Uses hybrid search (BM25 + vector similarity) via Weaviate
- Integrates with Slack API to generate message permalinks

**Data Indexing Pipeline**
- `weaviate_indexer.py`: Core indexing class that embeds documents into Weaviate
- `index_slack.py`, `index_wiki.py`, `index_web.py`: Source-specific indexers
- Uses OpenAI `text-embedding-3-large` model for embeddings
- Stores documents in Weaviate with structured schema (title, link, source, text)

**State Management (`state.py`)**
- Centralizes Streamlit session state initialization
- Manages model selection, search parameters, and UI persistence
- Handles OpenAI model discovery and caching

**Slack Integration**
- `slack_app.py`: Slack bot application for responding to queries within Slack workspace
- `slack_scrape/`: Directory containing incremental scraping and indexing with flexible intervals

### Data Flow

1. **Indexing**: Documents → OpenAI Embeddings → Weaviate Vector Store (with class prefix like "Janelia_Node")
2. **Search**: User Query → Vector Search + BM25 → LlamaIndex Query Engine → OpenAI GPT → Formatted Response with Sources
3. **UI**: Streamlit pages with persistent state across navigation

### Key Configuration

**Environment Variables Required:**
- `OPENAI_API_KEY`: For embeddings and chat completion
- `SLACK_TOKEN`: For generating Slack message permalinks  
- `CONFLUENCE_TOKEN`: For wiki data download

**Search Parameters (configurable in Settings):**
- Search Alpha: Balance between keyword (BM25) and vector search (0-100)
- Num Results: Number of source documents fed to GPT (affects latency)
- Temperature: GPT creativity control (keep low to avoid hallucination)
- Model: Selectable OpenAI model (default: gpt-4o)

### Database Schema

Weaviate stores documents with this schema:
- `ref_doc_id`: Document reference ID
- `text`: Full document text (searchable)
- `title`: Document title
- `link`: Source URL
- `source`: Data source type (slack/wiki/web)
- `scraped_at`: Unix timestamp when the document was scraped

## Testing and Quality

**Built-in Testing Framework**: The system includes comprehensive testing in `unit_tests/` with custom test cases for specific domain knowledge validation. Tests use `deepeval` for LLM response evaluation and can be run via the "Unit Tests" page in the Streamlit interface.

**Test Structure**: `unit_tests/test_cases.py` contains predefined test cases with expected responses for Janelia-specific queries. Custom tests check for specific content (email addresses, wiki links, people names) in responses and sources.

**Evaluation Frameworks**: Multiple evaluation approaches available in `eval/` directory:
- DeepEval: Hallucination and answer relevancy metrics
- RAGAS: RAG system evaluation 
- HuggingFace: Alternative evaluation methods

## Development Environment

**Dependency Management**: This project uses `pixi` for dependency management instead of pip/conda. All Python commands should be prefixed with `pixi run` or run within `pixi shell`.

**Docker Development**: Use `docker-compose-dev.yaml` for testing locally built containers before pushing to GitHub Container Registry. The regular `docker-compose.yaml` uses published images.

**Port Configuration**: Weaviate runs on port 8080 inside containers but is exposed as 8777 on the host. When connecting from local development, use `http://localhost:8777`. When containers communicate internally, they use `http://weaviate:8080`.

## Data Sources

- **Slack**: Export files processed to extract messages with threading context
- **Wiki**: Confluence pages downloaded via API and processed  
- **Web**: Janelia.org scraped using Scrapy spider
- **Survey**: User feedback collected and stored in separate Weaviate class

## Key Implementation Details

**Hybrid Search**: Combines BM25 (keyword) and vector search with configurable alpha parameter (0-100) for balancing approaches. Uses Reciprocal Rank Fusion for result re-ranking.

**Multi-Modal Pages**: Streamlit interface includes multiple pages:
- Main search (`1_🔍_Search.py`)
- Settings configuration (`pages/2_⚙️_Settings.py`) 
- Unit testing interface (`pages/5_🧪_Unit_Tests.py`)
- Survey response management (`pages/4_📋_Survey_Responses.py`)

**Container Architecture**: Separate containers for web app, Slack bot, and Weaviate database with internal/external port mapping (8080 internal, 8777 external for Weaviate).