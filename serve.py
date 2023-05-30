#!/usr/bin/env python

import os
import re
import sys
import argparse
import textwrap
import logging
import warnings
from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LLMPredictor, PromptHelper, ServiceContext, LangchainEmbedding, GPTVectorStoreIndex, ResponseSynthesizer
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.vector_stores.types import VectorStoreQueryMode

import weaviate
import streamlit as st
from slack_sdk import WebClient


warnings.simplefilter("ignore", ResourceWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger('llama_index').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Constants
TOP_N = 3
MAX_INPUT_SIZE = 4096
NUM_OUTPUT = 256
MAX_CHUNK_OVERLAP = 20
SURVEY_CLASS = "SurveyResponses"

FAQ = f"""
# FAQ

## How does JaneliaGPT work?
Content from the Janelia-Software Slack and Janelia Wiki are translated into semantic vectors using OpenAI's embedding API 
and stored in a vector database. Your query is embedded as well and used to search the database for content that is 
semantically related to your query. The GPT language model tries to answer your question using the top {TOP_N} results.

## Why is the answer unrelated to my question?
At the moment this is just a proof of concept. It works brilliantly for some questions and fails spectacularly for others.
Please use the survey buttons to record your experience so that we can use the results to improve the search results in future iterations. 

## Why can't I find something that was posted recently?
Source data was downloaded on May 19, 2023. If the search proves useful, we can implement automated downloading and incremental indexing. 

## Where is the source code?
[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/JaneliaSciComp/gpt-semantic-search)
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


@st.cache_data
def get_message_link(_slack_client, channel, ts):
    res = _slack_client.chat_getPermalink(channel=channel, message_ts=ts)
    if res['ok']:
        return res['permalink']
    

@st.cache_resource
def get_weaviate_client(weaviate_url):

    client = weaviate.Client(weaviate_url)

    if not client.is_live():
        raise Exception(f"Weaviate is not live at {weaviate_url}")

    if not client.is_live():
        raise Exception(f"Weaviate is not ready at {weaviate_url}")

    return client


@st.cache_resource
def get_slack_client():
    slack_client = WebClient(token=os.environ.get('SLACK_TOKEN'))
    res = slack_client.api_test()
    if not res["ok"]:
        logger.error(f"Error initializing Slack API: {res['error']}")
        sys.exit(1)
    
    return slack_client


@st.cache_resource
def get_query_engine(_weaviate_client, model, class_prefix):

    # Based on experimentation, gpt-3.5-turbo does not do well with Slack documents, using text-curie-001 for now. 
    llm = ChatOpenAI(temperature=0.7, model_name=model)
    llm_predictor = LLMPredictor(llm=llm)
    embed_model = LangchainEmbedding(OpenAIEmbeddings())
    prompt_helper = PromptHelper(MAX_INPUT_SIZE, NUM_OUTPUT, MAX_CHUNK_OVERLAP)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)
    vector_store = WeaviateVectorStore(weaviate_client=_weaviate_client, class_prefix=class_prefix)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex([], storage_context=storage_context, service_context=service_context)


    # configure retriever
    retriever = VectorIndexRetriever(
        index,
        similarity_top_k=TOP_N,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        alpha=0.75,
    )

    # configure response synthesizer
    synth = ResponseSynthesizer.from_args()

    # construct query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synth,
    )

    return query_engine


@st.cache_data
def get_response(_query_engine, _slack_client, query):

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

        if source == 'slack':
            channel_id = extra_info['channel']
            ts = extra_info['ts']
            msg += f"* Slack: [{text}]({get_message_link(_slack_client, channel_id, ts)})\n"

        elif source == 'wiki':
            msg += f"* Wiki: [{extra_info['title']}]({extra_info['link']})\n"
    
    return msg


def main():
    
    parser = argparse.ArgumentParser(description='Web service for semantic search using Weaviate and OpenAI')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8080", help='Weaviate database URL')
    parser.add_argument('-c', '--class-prefix', type=str, default="Janelia", help='Class prefix in Weaviate. The full class name will be "<prefix>_Node".')
    parser.add_argument('-m', '--model', type=str, default="text-curie-001", help='OpenAI model to use for query completion.')
    args = parser.parse_args()

    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .appview-container .main .block-container {
            padding-top: 1em;
        }
        div.css-1544g2n {
            padding-top: 1em;
        }
        [data-testid="stSidebar"] {
            font-size: 0.8em;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.5em;
        }
        [data-testid="stSidebar"] h2 {
            font-size: 1.25em;
        }
        [data-testid="stSidebar"] p {
            font-size: 1em;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(FAQ)

    if 'survey_complete' not in st.session_state:
        st.session_state.survey_complete = True

    if 'query' not in st.session_state:
        st.session_state.query = ""
        
    weaviate_client = get_weaviate_client(args.weaviate_url)
    query_engine = get_query_engine(weaviate_client, args.model, args.class_prefix)
    slack_client = get_slack_client()
    
    st.title("Ask JaneliaGPT")
    query = st.text_input("What would you like to ask?", '')
    if st.button("Submit") or (query and query == st.session_state.query):
        logger.info(f"Query: {query}")
        try:
            msg = get_response(query_engine, slack_client, query)    
            st.success(msg)
            logger.info(f"Response: {msg}")
        except Exception as e:
            msg = f"An error occurred: {e}"
            st.error(msg)
            logger.error(f"Response: {msg}")
        
        if st.session_state.query != query:
            # First time rendering this query/response, record it and ask for survey
            st.session_state.db_id = record_log(weaviate_client, query, msg)
            st.session_state.query = query
            st.session_state.survey_complete = False

    
    def survey_click(survey_response):
        
        st.session_state.survey = survey_response
        st.session_state.survey_complete = True

        create_survey_schema(weaviate_client)

        db_id = st.session_state.db_id
        record_survey(weaviate_client, db_id, survey_response)
        logger.info(f"Logged survey response: {survey_response}")
        del st.session_state['survey']


    if not st.session_state.survey_complete:
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


        with st.form("survey_form"):
            st.markdown("Was your question answered?")
            col1, col2 = st.columns([1,1])
            with col1:
                st.form_submit_button("Yes", on_click=survey_click, args=('Yes', ))
            with col2:
                st.form_submit_button("No", on_click=survey_click, args=('No', ))
    

if __name__ == '__main__':
    main()