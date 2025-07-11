# gpt-semantic-search

Semantic search service for Janelia resources using OpenAI GPT models. This repository contains tools for indexing various Janelia resources (website, wiki, Slack) into a [Weaviate](https://weaviate.io/) vector database, and a simple web-UI built with [Streamlit](https://streamlit.io/) which allows users to query the indexed data set using a ChatGPT-style interface.

## Running

This service requires [Docker](https://docs.docker.com/get-docker/) to be installed. To run, simply clone this repo and start the Compose deployment:

    docker compose up -d

This will start the Weaviate vector database, Streamlit webapp, and nginx reverse proxy. You can then access the webapp at:
- **Local development**: http://localhost:8501 (direct Streamlit access)
- **Production with nginx**: https://localhost:443 (SSL with nginx proxy)

For production deployment on the cluster, the service will be available at https://search.int.janelia.org.

### SSL Certificate Setup

If using nginx with SSL, you'll need to generate a Diffie-Hellman parameters file:

    openssl dhparam -out docker/include/dhparam.pem 4096

Set the required environment variables for SSL certificates:

    export CERT_DIR=/path/to/ssl/certificates

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

1. Confluence Wiki - run `pixi run python -m scraping.wiki.download_wiki` to download the latest wiki pages to `./data/wiki`
2. Slack - export data from Slack using their [export tool](https://slack.com/help/articles/201658943-Export-your-workspace-data).
3. Janelia.org - run the web crawling spider with `pixi run scrapy runspider scraping/web/spider.py`

### Run indexing

Add a wiki export:

    pixi run python -m indexing.wiki.index_wiki -i ./data/wiki -c Janelia

Add the janelia.org web site:

    pixi run python -m indexing.web.index_web -i ./data/janelia.org -c Janelia

Add a Slack export:

    pixi run python -m indexing.slack.index_slack -i ./data/slack/slack_export_Janelia-Software_ALL -c Janelia


### Start semantic search webapp

To run the webapp in dev mode:

    pixi run streamlit run 1_🔍_Search.py -- -w http://localhost:8777
    
## Development Notes

### Getting notebooks to work in VS Code

You need to install a Jupyter kernel that point to the virtualenv:

    pixi add jupyterlab pixi-kernel  
    pixi add ipykernel 

And then to open in JupyterLab: 
    
    pixi run jupyter lab

### Rebuild container

Build from this directory (setting a version number instead of "latest"):

    docker build --no-cache . -f docker/Dockerfile -t ghcr.io/janeliascicomp/gpt-semantic-search-web:latest

Then push:

    docker push ghcr.io/janeliascicomp/gpt-semantic-search-web:latest

Once the upload is done, remember to update the version number in `docker-compose.yaml`.

To rebuild the Slack bot:

    docker build --no-cache . -f docker/Dockerfile_slack -t ghcr.io/janeliascicomp/gpt-semantic-search-slack-bot:latest

### Multi-arch builds

To build the Slackbot container for both Linux and Mac:

    export VERSION=latest
    docker buildx build --build-arg $VERSION --platform linux/arm64,linux/amd64 --tag ghcr.io/janeliascicomp/gpt-semantic-search-slack-bot:$VERSION -f docker/Dockerfile_slack .

### Production Deployment

For production deployment with nginx reverse proxy:

```bash
# Deploy all services including nginx
docker compose up -d

# Or deploy just nginx configuration for testing
cd docker/
docker compose up -d
```


## Slack Scraping

### Required Environment Variable

For Slack scraping, you need a `SCRAPING_SLACK_USER_TOKEN` environment variable. Add this to your shell profile (e.g., `~/.bashrc`, `~/.zshrc`):

    export SCRAPING_SLACK_USER_TOKEN="xoxp-your-user-token-here"

#### Automated scraping and indexing

**Set up cron jobs:**

```bash
# Daily scraping - automatically continues from last successful run
0 0 * * * cd /path/to/gpt-semantic-search && pixi run python -m scraping.slack.slack_incremental_scraper >> logs/slack_scraper.log 2>&1

# Daily indexing - smart discovery of new data
0 3 * * * cd /path/to/gpt-semantic-search && pixi run python -m indexing.slack.slack_incremental_indexer >> logs/slack_indexer.log 2>&1
```
   
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


