# gpt-semantic-search

Semantic search service for Janelia resources using OpenAI GPT models.

Largely based on [this tutorial](https://alphasec.io/query-your-own-documents-with-llamaindex-and-langchain/).

## Running

This service requires [Docker](https://docs.docker.com/get-docker/) to be installed. To run, simply clone this repo and start the Compose deployment:

    docker compose up -d

This will start both the Weaviate vector database, and the Streamlit webapp. You can then access the webapp at [http://localhost:8501]().


## Development

### Install dependencies

Create a virtualenv and install the dependencies:

    $ virtualenv <env_name>
    $ source <env_name>/bin/activate
    (<env_name>)$ pip install -r path/to/requirements.txt

### Launch Weaviate database

You can run just Weaviate as follows:

    docker compose up weaviate -d

You can verify that Weaviate is running by opening [http://localhost:8080]() in your browser.

### Set up tokens

To index data or run a search, you must have an `OPENAI_API_KEY` set in your environment. You can obtain one by logging into the OpenAI web app and navigating to [API keys](https://platform.openai.com/account/api-keys).

In order for the search webapp to generate links back to Slack messages, you must have a `SLACK_TOKEN` set in your environment. You can [generate one here](https://api.slack.com/tutorials/tracks/getting-a-token).

For running the Wiki download, you must have a `CONFLUENCE_TOKEN` in your environment. You can create one by logging into the wiki and selecting your profile in the upper right, then selecting "Personal Access Tokens". If you are at Janelia, [click here](https://wikis.janelia.org/plugins/personalaccesstokens/usertokens.action).

### Download data sources

If you are at Janelia you can experiment easily by copying the data sources from shared storage on NRS:

    mkdir ./data
    copy -R /nrs/scicompsoft/rokicki/semantic-search/wiki ./data

If you want to download the latest data, you can use the [DownloadConfluence.ipynb](notebooks/DownloadConfluence.ipynb) notebook to download the wiki for yourself. You can export data from Slack using their [export tool](https://slack.com/help/articles/201658943-Export-your-workspace-data).

### Run indexing

Index a Slack export to the Janelia class in Weaviate:

    ./index_slack.py -i ./data/slack/slack_export_Janelia-Software_ALL -c Janelia

Add a wiki export:

    ./index_wiki.py -i ./data/wiki -c Janelia

### Start semantic search webapp

    streamlit run ./serve.py

If you want to pass arguments, such as a different class prefix, use two dashes:

    streamlit run ./serve.py -- -c MyPrefix

## Development Notes

### Getting notebooks to work in VS Code

You need to install a Jupyter kernel that point to the virtualenv:

    python3 -m ipykernel install --user --name=env

And then select the env as the Python Interpreter for the notebook.

### Rebuild container

Build from this directory (setting a version number instead of latest):

    docker build . -t registry.int.janelia.org/scsw/semantic-search:latest

Then push:

    docker push registry.int.janelia.org/scsw/semantic-search:latest

Once the upload is done, remember to update the version number in `docker-compose.yaml`.

### Update requirements.txt

Run this in the venv:

    pip3 freeze > requirements.txt


