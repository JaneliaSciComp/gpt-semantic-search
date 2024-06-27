#!/usr/bin/env python

import os
import re
import sys
import argparse
import textwrap
import logging
import warnings
from typing import Dict, List

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import PromptHelper, GPTVectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode

import weaviate
import streamlit as st

st.set_page_config(page_title="JaneliaGPT", page_icon="🐛")

from state import init_state
init_state()

warnings.simplefilter("ignore", ResourceWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger('llama_index').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants
EMBED_MODEL_NAME="text-embedding-3-large"
CONTEXT_WINDOW = 4096
NUM_OUTPUT = 256
CHUNK_OVERLAP_RATIO = 0.1
SURVEY_CLASS = "SurveyResponses"

SIDEBAR_DESC = """
JaneliaGPT uses OpenAI models to index various data sources in a vector database for searching. 
Currently the following sources are indexed:
* Janelia.org
* Janelia-Software Slack Workspace
* Janelia Wiki (spaces 'SCSW', 'SCS', and 'ScientificComputing')
"""

NODE_SCHEMA: List[Dict] = [
    {
        "dataType": ["text"],
        "description": "User query",
        "name": "query"
    },
    {
        "dataType": ["text"],
        "description": "GPT response",
        "name": "response"
    },
    {
        "dataType": ["text"],
        "description": "Survey response",
        "name": "survey",
    },
]

def create_survey_schema(weaviate_client) -> None:
    """Create schema."""
    # first check if schema exists
    schema = weaviate_client.schema.get()
    classes = schema["classes"]
    existing_class_names = {c["class"] for c in classes}
    # if schema already exists, don't create
    if SURVEY_CLASS in existing_class_names:
        return

    properties = NODE_SCHEMA
    class_obj = {
        "class": SURVEY_CLASS,  # <= note the capital "A".
        "description": f"Class for survey responses",
        "properties": properties,
    }
    weaviate_client.schema.create_class(class_obj)


def record_log(weaviate_client, query, response):
    metadata = {
        "query": query,
        "response": response,
        'survey': 'Unknown'
    }
    return weaviate_client.data_object.create(metadata, SURVEY_CLASS)


def record_survey(weaviate_client, db_id, survey):
    metadata = {
        "survey": survey,
    }
    weaviate_client.data_object.update(metadata, SURVEY_CLASS, db_id)


def get_unique_nodes(nodes):
    docs_ids = set()
    unique_nodes = list()
    for node in nodes:
        if node.node.ref_doc_id not in docs_ids:
            docs_ids.add(node.node.ref_doc_id)
            unique_nodes.append(node)
    return unique_nodes


def escape_text(text):
    text = re.sub("<", "&lt;", text)
    text = re.sub(">", "&gt;", text)
    text = re.sub("([_#])", "\\\1", text)
    return text



    

@st.cache_resource
def get_weaviate_client(weaviate_url):

    client = weaviate.Client(weaviate_url)

    if not client.is_live():
        raise Exception(f"Weaviate is not live at {weaviate_url}")

    if not client.is_live():
        raise Exception(f"Weaviate is not ready at {weaviate_url}")

    return client





def get_query_engine(_weaviate_client):

    model = st.session_state["model"]
    class_prefix = st.session_state["class_prefix"]
    temperature = st.session_state["temperature"] / 100.0
    search_alpha = st.session_state["search_alpha"] / 100.0
    num_results = st.session_state["num_results"]

    logger.info("Getting query engine with parameters:")
    logger.info(f"  model: {model}")
    logger.info(f"  class_prefix: {class_prefix}")
    logger.info(f"  temperature: {temperature}")
    logger.info(f"  search_alpha: {search_alpha}")
    logger.info(f"  num_results: {num_results}")

    llm = OpenAI(model=model, temperature=temperature)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL_NAME)
    prompt_helper = PromptHelper(CONTEXT_WINDOW, NUM_OUTPUT, CHUNK_OVERLAP_RATIO)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.prompt_helper = prompt_helper

    vector_store = WeaviateVectorStore(weaviate_client=_weaviate_client, class_prefix=class_prefix)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex([], storage_context=storage_context)

    # configure retriever
    retriever = VectorIndexRetriever(
        index,
        similarity_top_k=num_results,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        alpha=search_alpha,
    )

    # construct query engine
    query_engine = RetrieverQueryEngine.from_args(retriever)

    return query_engine


def get_response(_query_engine, query):

    # Escape certain characters which the 
    query = re.sub("\"", "", query)

    response = _query_engine.query(query)

    msg = f"{response.response}\n\nSources:\n\n"
    for node in get_unique_nodes(response.source_nodes):
        extra_info = node.node.extra_info
        text = node.node.text

        text = re.sub("\n+", " ", text)
        text = textwrap.shorten(text, width=100, placeholder="...")
        text = escape_text(text)

        source = extra_info['source']

        
        msg += f"* {source}: [{extra_info['title']}]({extra_info['link']})\n"

    return msg


@st.cache_data
def get_cached_response(_query_engine, query):
    return get_response(_query_engine, query)


parser = argparse.ArgumentParser(description='Web service for semantic search using Weaviate and OpenAI')
parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8777", help='Weaviate database URL')
args = parser.parse_args()

st.sidebar.markdown(SIDEBAR_DESC)

if 'survey_complete' not in st.session_state:
    st.session_state.survey_complete = True

if 'query' not in st.session_state:
    st.session_state.query = ""

weaviate_client = get_weaviate_client(args.weaviate_url)

st.title("Ask JaneliaGPT")
query = st.text_input("What would you like to ask?", '', key="query")


#If query is filled in (which occurs when enter key is pressed) or the submit button is clicked
if query or st.button("Submit"):
    logger.info(f"Query: {query}")
    try:
        query_engine = get_query_engine(weaviate_client)
        msg = get_response(query_engine, query)  
        st.session_state.db_id = record_log(weaviate_client, query, msg)
        st.session_state.survey_complete = False
        st.session_state.response = msg
        st.session_state.response_error = False
        logger.info(f"Response saved as {st.session_state.db_id}: {msg}")
        st.success(msg)
    except Exception as e:
        msg = f"An error occurred: {e}"
        st.session_state.response = msg
        st.session_state.response_error = True
        logger.exception(msg)
        st.error(msg)

elif st.session_state.response:
    # Re-render the saved response
    if st.session_state.response_error:
        st.error(st.session_state.response)
    else:
        st.success(st.session_state.response)


def survey_click(survey_response):

    st.session_state.survey = survey_response
    st.session_state.survey_complete = True

    create_survey_schema(weaviate_client)

    db_id = st.session_state.db_id
    record_survey(weaviate_client, db_id, survey_response)
    logger.info(f"Logged survey response: {survey_response}")
    del st.session_state['survey']


if st.session_state.response and not st.session_state.survey_complete:
    st.markdown(
        """
        <style>
            div[data-testid="column"]:nth-of-type(1)
            {
                text-align: end;
            } 
        </style>
        """,unsafe_allow_html=True
    )

    with st.form(key="survey_form"):
        st.markdown("Was your question answered?")
        col1, col2 = st.columns([1,1])
        with col1:
            st.form_submit_button("Yes", on_click=survey_click, args=('Yes', ))
        with col2:
            st.form_submit_button("No", on_click=survey_click, args=('No', ))
