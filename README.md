# gpt-semantic-search

Semantic search service for Janelia resources using OpenAI GPT models. This repository contains tools for indexing various Janelia resources (website, wiki, Slack) into a [Weaviate](https://weaviate.io/) vector database, and a simple web-UI built with [Streamlit](https://streamlit.io/) which allows users to query the indexed data set using a ChatGPT-style interface.

## Running

This service requires [Docker](https://docs.docker.com/get-docker/) to be installed. To run, simply clone this repo and start the Compose deployment:

    docker compose up -d

This will start both the Weaviate vector database, and the Streamlit webapp. You can then access the webapp at http://localhost:8501.

## Development

### Install dependencies

Install pixi if you haven't already:

    curl -fsSL https://pixi.sh/install.sh | bash

Install dependencies and activate environment:

    pixi install
    pixi shell

### Launch Weaviate database

You can run just Weaviate as follows:

    docker compose up weaviate -d

You can verify that Weaviate is running by opening [http://localhost:8777]() in your browser.

### Set up tokens

To index data or run a search, you must have an `OPENAI_API_KEY` set in your environment. You can obtain one by logging into the OpenAI web app and navigating to [API keys](https://platform.openai.com/account/api-keys).

In order for the search webapp to generate links back to Slack messages, you must have a `SLACK_TOKEN` set in your environment. You can [generate one here](https://api.slack.com/tutorials/tracks/getting-a-token).

For running the Wiki download, you must have a `CONFLUENCE_TOKEN` in your environment. You can create one by logging into the wiki and selecting your profile in the upper right, then selecting "Personal Access Tokens". If you are at Janelia, [click here](https://wikis.janelia.org/plugins/personalaccesstokens/usertokens.action).

### Download data sources

If you are at Janelia you can get data sources from shared storage:

    mkdir ./data
    cp -R /nrs/scicompsoft/rokicki/semantic-search/data ./data

If you want to download the latest data from each source:

1. Confluence Wiki - run `pixi run python download_wiki.py` to download the latest wiki pages to `./data/wiki`
2. Slack - export data from Slack using their [export tool](https://slack.com/help/articles/201658943-Export-your-workspace-data).
3. Janelia.org - run the web crawling spider with `pixi run scrapy runspider spider.py`

### Run indexing

Index a Slack export to the Janelia class in Weaviate:

    pixi run ./index_slack.py -i ./data/slack/slack_export_Janelia-Software_ALL -c Janelia

    pixi run ./index_slack.py -i ./data/slack/janelia-software/slack_to_2023-05-18 -c Janelia

Add a wiki export:

    pixi run ./index_wiki.py -i ./data/wiki -c Janelia

Add the janelia.org web site:

    pixi run ./index_web.py -i ./data/janelia.org -c Janelia


### Start semantic search webapp

To run the webapp in dev mode:

    pixi run streamlit run 1_🔍_Search.py -- -w http://localhost:8777
    
    pixi run streamlit run No_Slack_Search.py -- -w http://localhost:8777

## Development Notes

### Getting notebooks to work in VS Code

You need to install a Jupyter kernel that point to the virtualenv:

    python3 -m ipykernel install --user --name=env

And then select the env as the Python Interpreter for the notebook.

### Rebuild container

Build from this directory (setting a version number instead of "latest"):

    docker build --no-cache . -t ghcr.io/janeliascicomp/gpt-semantic-search-web:latest

Then push:

    docker push ghcr.io/janeliascicomp/gpt-semantic-search-web:latest

Once the upload is done, remember to update the version number in `docker-compose.yaml`.

To rebuild the Slack bot:

    docker build --no-cache . -f Dockerfile_slack -t ghcr.io/janeliascicomp/gpt-semantic-search-slack-bot:latest

### Multi-arch builds

To build the Slackbot container for both Linux and Mac:

    export VERSION=latest
    docker buildx build --build-arg $VERSION --platform linux/arm64,linux/amd64 --tag ghcr.io/janeliascicomp/gpt-semantic-search-slack-bot:$VERSION -f Dockerfile_slack .


## Slack Scraping

### Required Environment Variable

For Slack scraping, you need a `SCRAPING_SLACK_USER_TOKEN` environment variable. Add this to your shell profile (e.g., `~/.bashrc`, `~/.zshrc`):

    export SCRAPING_SLACK_USER_TOKEN="xoxp-your-user-token-here"

### One-time historical scraping (for initial setup)

    cd slack_scrape
    pixi run python slack_past_scraper.py

### Daily automated scraping and indexing (recommended)

The system uses a hybrid approach with separate processes for scraping and indexing:

1. **Daily Scraper** (midnight): Fast, reliable message collection
2. **Daily Indexer** (3 AM): Process and index messages into Weaviate

#### Setting up the cron jobs

1. **Open your crontab for editing:**
   ```bash
   crontab -e
   ```

2. **Add both cron jobs** (adjust paths to match your installation):
   ```bash
   # Daily scraping at midnight
   0 * * * * cd /path/to/gpt-semantic-search/slack_scrape && pixi run python slack_daily_scraper.py >> logs/cron.log 2>&1
   
   # Daily indexing at 1 AM
   0 1 * * * cd /path/to/gpt-semantic-search/slack_scrape && pixi run python slack_daily_indexer.py >> logs/cron.log 2>&1
   ```

3. **Find your correct paths:**
   - Project path: `pwd` (when in the project directory)
   - Pixi path: `which pixi`

4. **Example with full paths:**
   ```bash
   # Daily scraping at midnight
   0 0 * * * cd /Users/username/gpt-semantic-search/slack_scrape && /Users/username/.pixi/bin/pixi run python slack_daily_scraper.py >> logs/cron.log 2>&1
   
   # Daily indexing at 3 AM
   0 3 * * * cd /Users/username/gpt-semantic-search/slack_scrape && /Users/username/.pixi/bin/pixi run python slack_daily_indexer.py >> logs/cron.log 2>&1
   ```

#### How it works

**Daily Scraper (12:00 AM):**
- Discovers all accessible public channels automatically
- Fetches messages from the last 24 hours (or since last run)
- Saves data to `../data/slack/{workspace}/`
- Fast and reliable (2-5 minutes)
- Logs to `logs/slack_daily_scraper_YYYYMMDD.log`

**Daily Indexer (3:00 AM):**
- Processes yesterday's scraped messages
- Creates searchable documents
- Indexes them into Weaviate database
- Makes messages searchable within 3-6 hours
- Logs to `logs/slack_daily_indexer_YYYYMMDD.log`

#### Configuration options

**Specify specific channels** (optional):
```bash
export SLACK_CHANNELS="general,development,support"
```

**Enable/disable indexing:**
```bash
export ENABLE_INDEXING="true"              # Enable automatic indexing
export WEAVIATE_URL="http://localhost:8777"  # Optional, defaults to this
export CLASS_PREFIX="Janelia"               # Optional, defaults to this
```

**Required for indexing:**
- `OPENAI_API_KEY` must be set for embedding generation

#### Benefits of hybrid approach

- **Reliability**: Scraping and indexing failures don't affect each other
- **Speed**: Daily scraping completes quickly (no embedding overhead)
- **Efficiency**: Better API usage and error recovery
- **Availability**: Messages searchable within 3-6 hours

### Update dependencies

To update pixi.toml with new dependencies, edit the file directly or use:

    pixi add <package-name>


## Future Directions

* Run search when user presses the RETURN key
* Add option to decrease `top_p` for more deterministic responses
* Ways to "correct" the model over time
    * Ability to remove (i.e. block) incorrect sources from the database
    * Weight more recent data more highly in the search results
* SlackBot
* Additional custom prompting
    * Focus answers on Janelia employees
    * Redirect to HughesHub if unable to answer a question


