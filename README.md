# gpt-semantic-search

Semantic search for Janelia resources using GPT models

## Getting started

### Install dependencies

Create a virtualenv and install the dependencies:

    $ virtualenv <env_name>
    $ source <env_name>/bin/activate
    (<env_name>)$ pip install -r path/to/requirements.txt

### Launch Weaviate database

This requires [Docker](https://docs.docker.com/get-docker/) to be installed.

    docker compose up -d

You can verify that Weaviate is running by opening [http://localhost:8080]() in your browser.

### Download data sources

For now, you can experiment easily by copying the wiki data source from shared storage on NRS:

    mkdir ./data
    copy -R /nrs/scicompsoft/rokicki/semantic-search/wiki ./data

If you want the latest data, you can use the [DownloadConfluence.ipynb](notebooks/DownloadConfluence.ipynb) notebook to download the wiki for yourself.

### Run indexing

Launch the notebook and work through the cells one by one:

    jupyter notebook LlamaIndexToWeaviate.ipynb

### Start semantic search webapp

    streamlit run ./serve.py

## Development Notes

### Getting notebooks to work in VS Code

You need to install a Jupyter kernel that point to the virtualenv:

    python3 -m ipykernel install --user --name=env

And then select the env as the Python Interpreter for the notebook.
