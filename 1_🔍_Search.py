#!/usr/bin/env python

import os
import re
import sys
import argparse
import textwrap
import logging
import warnings
from typing import Dict, List
import time
from ollama_client import SimpleOllamaAPI

import weaviate
import streamlit as st
from slack_sdk import WebClient

st.set_page_config(page_title="JaneliaGPT", page_icon="🔍")

from state import init_state
init_state()

warnings.simplefilter("ignore", ResourceWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger('llama_index').setLevel(logging.DEBUG)
logging.getLogger('ollama').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants
EMBED_MODEL_NAME="bge-m3:567m"
CONTEXT_WINDOW = 4096
NUM_OUTPUT = 256
CHUNK_OVERLAP_RATIO = 0.1
SURVEY_CLASS = "SurveyResponses"

SIDEBAR_DESC = """
JaneliaGPT uses Ollama local models to index various data sources in a vector database for searching. 
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

    return client


@st.cache_resource
def get_slack_client():
    slack_client = WebClient(token=os.environ.get('SLACK_TOKEN'))
    res = slack_client.api_test()
    if not res["ok"]:
        logger.error(f"Error initializing Slack API: {res['error']}")
        sys.exit(1)

    return slack_client


def search_and_generate(_weaviate_client, query):

    class_prefix = st.session_state["class_prefix"]
    temperature = st.session_state["temperature"] / 100.0
    num_results = st.session_state["num_results"]

    logger.info("Searching with parameters:")
    logger.info(f"  class_prefix: {class_prefix}")
    logger.info(f"  temperature: {temperature}")
    logger.info(f"  num_results: {num_results}")

    ollama = SimpleOllamaAPI()
    class_name = f"{class_prefix}_Node"
    
    # Get query embedding and search Weaviate
    query_embedding = ollama.get_embedding(query)
    
    logger.info(f"Searching Weaviate class '{class_name}' with {len(query_embedding)} dimensional vector")
    
    result = (
        _weaviate_client.query
        .get(class_name, ["text", "title", "link", "source", "ref_doc_id"])
        .with_near_vector({"vector": query_embedding})
        .with_limit(num_results)
        .do()
    )
    
    logger.debug(f"Weaviate search result: {result}")
    
    if not result.get("data", {}).get("Get", {}).get(class_name):
        logger.warning(f"No search results found for class '{class_name}'")
        logger.info("This might be because:")
        logger.info("1. No documents are indexed in this class")
        logger.info("2. Documents were indexed without vector embeddings")
        logger.info("3. The embedding dimensions don't match")
        
        # Check if class exists and has data
        try:
            schema = _weaviate_client.schema.get(class_name)
            logger.info(f"Class {class_name} exists in schema")
            
            # Get a count of objects in the class
            count_result = _weaviate_client.query.aggregate(class_name).with_meta_count().do()
            count = count_result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", 0)
            logger.info(f"Class {class_name} contains {count} objects")
            
            if count > 0:
                logger.warning("Objects exist but vector search returned no results - likely missing embeddings")
        except Exception as e:
            logger.error(f"Error checking class status: {str(e)}")
        
        return None, []
    
    search_results = result["data"]["Get"][class_name]
    
    # Build context for LLM
    context = "Context:\n"
    for i, result in enumerate(search_results, 1):
        text = result.get("text", "")[:500]  # Limit context size
        title = result.get("title", "Untitled")
        context += f"{i}. {title}: {text}\n\n"
    
    # Generate response with Ollama
    prompt = f"""Based on the following context, answer the user's question: "{query}"

{context}

Provide a helpful and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, say so."""
    
    response_text = ollama.generate(prompt, temperature=temperature)
    
    return response_text, search_results


def format_response_with_sources(response_text, search_results, query):
    # Original response formatting logic
    if not response_text:
        return f"No results found for: '{query}'"
    
    formatted_response = response_text
    
    if search_results:
        formatted_response += "\n\n**Sources:**\n\n"
        
        for result in search_results:
            text = result.get("text", "")
            text = re.sub(r"\n+", " ", text)
            text = textwrap.shorten(text, width=100, placeholder="...")
            text = escape_text(text)
            
            source = result.get("source", "Unknown")
            title = result.get("title", "Untitled")
            link = result.get("link", "")
            
            if link:
                formatted_response += f"* {source}: {title}\n  Link: {link}\n  {text}\n\n---\n\n"
            else:
                formatted_response += f"* {source}: {title}\n  {text}\n\n---\n\n"
    
    return formatted_response


def old_get_query_engine(_weaviate_client):

    model = st.session_state["model"]
    class_prefix = st.session_state["class_prefix"]
    temperature = st.session_state["temperature"] / 100.0
    search_alpha = st.session_state["search_alpha"] / 100.0
    num_results = st.session_state["num_results"]
    hyde_enabled = st.session_state.get("hyde_enabled", False)

    logger.info("Getting query engine with parameters:")
    logger.info(f"  model: {model}")
    logger.info(f"  class_prefix: {class_prefix}")
    logger.info(f"  temperature: {temperature}")
    logger.info(f"  search_alpha: {search_alpha}")
    logger.info(f"  num_results: {num_results}")
    logger.info(f"  hyde_enabled: {hyde_enabled} (type: {type(hyde_enabled)})")
    logger.info(f"  session_state.hyde_enabled: {st.session_state.get('hyde_enabled', 'NOT_SET')}")

    ollama = SimpleOllamaAPI()
    
    # This is now simplified - just return the ollama instance
    return ollama


def get_query_engine(_weaviate_client):
    # Legacy function - now just returns ollama
    return old_get_query_engine(_weaviate_client)


def old_retriever_method(_weaviate_client):
    # Old code moved here for reference
    # configure retriever
    retriever = VectorIndexRetriever(
        index,
        similarity_top_k=num_results,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        alpha=search_alpha,
    )

    # construct query engine
    query_engine = RetrieverQueryEngine.from_args(retriever)
    
    # Apply HyDE transformation if enabled
    if hyde_enabled:
        try:
            hyde_transform = HyDEQueryTransform(include_original=True)
            query_engine = TransformQueryEngine(query_engine, hyde_transform)
            logger.info("✓ HyDE query transformation applied successfully")
        except Exception as e:
            logger.warning(f"✗ Failed to apply HyDE transformation: {e}. Falling back to regular query engine.")
    else:
        logger.info("✓ Using regular query engine (HyDE disabled)")

    return query_engine


def get_response(_query_engine, _slack_client, query):
    # Clean query
    clean_query = re.sub('"', "", query)
    
    # Use new search and generate function
    response_text, search_results = search_and_generate(weaviate_client, clean_query)
    
    if not response_text:
        return f"No results found for: '{query}'"
    
    msg = f"{response_text}\n\n**Sources:**\n\n"
    
    for result in search_results:
        text = result.get("text", "")
        text = re.sub(r"\n+", " ", text)
        text = textwrap.shorten(text, width=100, placeholder="...")
        text = escape_text(text)
        
        source = result.get("source", "Unknown")
        title = result.get("title", "Untitled")
        link = result.get("link", "")
        
        if source.lower() == 'slack' and 'channel' in result and 'ts' in result:
            channel_id = result['channel']
            ts = result['ts']
            msg += f"* {source}: [{text}]({get_message_link(_slack_client, channel_id, ts)})\n"
        elif link:
            msg += f"* {source}: [{title}]({link})\n"
        else:
            msg += f"* {source}: {title}\n  {text}\n\n"
    
    return msg

parser = argparse.ArgumentParser(description='Web service for semantic search using Weaviate and Ollama')
parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8080", help='Weaviate database URL')
args = parser.parse_args()

weaviate_client = get_weaviate_client(args.weaviate_url)

st.sidebar.markdown(SIDEBAR_DESC)

# Show HyDE status in sidebar
if st.session_state.get("hyde_enabled", False):
    st.sidebar.success("🔬 HyDE: Enabled")
else:
    st.sidebar.info("🔬 HyDE: Disabled")

st.title("Ask JaneliaGPT")
query = st.text_input("What would you like to ask?", '', key="query")


is_new_query = query and query != st.session_state.last_processed_query

if is_new_query or st.button("Submit"):
    if query:  
        hyde_enabled = st.session_state.get("hyde_enabled", False)
        logger.info(f"Query: {query}")
        logger.info(f"HyDE enabled: {hyde_enabled}")
        
        start_time = time.time()
        
        try:
            query_engine = get_query_engine(weaviate_client)
            slack_client = get_slack_client()
            
            # Use the cached response function to avoid regeneration
            msg = get_response(query_engine, slack_client, query)
            
            end_time = time.time()
            response_time = end_time - start_time
            logger.info(f"Query processed in {response_time:.2f} seconds (HyDE: {hyde_enabled})")
            
            # Only create a new log entry if this is truly a new query
            if query != st.session_state.last_processed_query:
                st.session_state.db_id = record_log(weaviate_client, query, msg)
                st.session_state.last_processed_query = query
                st.session_state.survey_complete = False
            
            st.session_state.response = msg
            st.session_state.response_error = False
            logger.info(f"Response saved as {st.session_state.db_id}: {msg}")
            st.success(msg)
        except Exception as e:
            msg = f"An error occurred: {e}"
            st.session_state.response = msg
            st.session_state.response_error = True
            st.session_state.last_processed_query = query
            logger.exception(msg)
            st.error(msg)

elif st.session_state.response:
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